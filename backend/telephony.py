"""Telephony ingress adapter: PSTN phone audio -> the same VAD/ASR/Conversation/TTS pipeline.

BACKEND_COMPLETION.md Sec3.3 recommends "Option A - managed media streams" as
the fastest way to bridge a real phone call into this backend: a SIP trunk
provider (Twilio Media Streams, Plivo Audio Streams, Exotel Voice Streaming,
...) answers the PSTN call and bridges its audio to a WebSocket you control.
This module is a code-complete adapter for exactly ONE concrete instance of
that option, modeled on Twilio Media Streams' publicly documented WebSocket
protocol (JSON text frames only):

    -> {"event": "connected", "protocol": "Call", "version": "1.0.0"}   (ignored)
    -> {"event": "start", "start": {"callSid": ..., "streamSid": ...,
                                     "mediaFormat": {"encoding": "audio/x-mulaw",
                                                      "sampleRate": 8000, "channels": 1},
                                     "customParameters": {...}}}
    -> {"event": "media", "media": {"payload": "<base64 mulaw audio>"}}   (repeated)
    -> {"event": "stop", "stop": {...}}
    <- {"event": "media", "streamSid": ..., "media": {"payload": "<base64 mulaw audio>"}}

IMPORTANT SCOPE NOTE: there is no real telephony vendor account/trunk in this
environment, so this cannot be live-tested end-to-end against a real phone
call - that is expected. What IS complete and unit-tested here is the entire
code path: auth, framing, u-law<->PCM transcoding, resampling, and wiring into
the exact same VAD -> ASR -> ConversationManager -> TTS orchestration that
backend/main.py's `/ws/audio` browser handler already uses (see
capture_browser_audio() there) - this module does not reinvent that
orchestration, it reuses it via the same app.state.* objects. Making a real
phone number ring through to this route is a deployment/vendor step (buy a
DID, configure a SIP trunk / TwiML `<Connect><Stream url="wss://.../ws/telephony">`
or the Plivo/Exotel equivalent to point its Media Streams webhook at this
route) that is outside what code alone can provide; the code itself is
complete and tested against synthetic payloads shaped exactly like the real
documented protocol above.

This module is deliberately self-contained (its own `APIRouter`, no edits to
main.py) so it can be wired in later with one `app.include_router(...)` line.
It expects the same `app.state.*` objects capture_browser_audio() already
depends on (asr, conversation, llm, tts, call_store, security, asr_semaphore,
tts_semaphore, settings) - it does not construct its own model instances, so
wiring this router into the existing FastAPI app's lifespan is enough to make
it work end-to-end with the real ASR/LLM/TTS objects already loaded there.

Codec/rate handling (BACKEND_COMPLETION.md Sec3.3): G.711 u-law at 8 kHz is
the telephony-standard codec, while VAD/ASR are hard-pinned to 16 kHz
(settings.py). `audioop.ulaw2lin`/`audioop.lin2ulaw` (stdlib) convert u-law
bytes <-> 16-bit linear PCM; `audio_transcode.resample_pcm` (also stdlib-free,
built this session) does the 8kHz<->16kHz conversion on either leg. Neither
direction touches backend/vad.py, backend/asr.py, backend/conversation.py,
backend/llm.py, or backend/tts.py - those are used exactly as they exist
today, unmodified.

Auth (BACKEND_COMPLETION.md Sec4): capture_browser_audio() checks a shared
secret via a WS query param before accept(), because a raw WebSocket client
cannot set a custom header on the handshake. Some Twilio Media Streams
deployments DO support a query string on the `<Stream url="wss://host/path?token=...">`
TwiML attribute, so the same pre-accept query-param check is tried first here
too. But not every SIP-trunk provider/deployment makes that easy to set up
(dashboards that only take a bare hostname, proxies that strip query strings,
...), so if no query param is presented at all, the connection is accepted
provisionally and the token is instead expected in the "start" event's
documented `start.customParameters` object (Twilio's TwiML supports
`<Parameter name="token" value="..."/>` inside `<Stream>`, which arrives
exactly there) - checked before any audio is processed. A query param that IS
present but wrong is still rejected before accept(), same as the browser
handler. Document/adjust this choice per whichever real vendor is eventually
wired in; both paths are exercised in backend/test_telephony.py.
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import suppress
import json
import logging
from uuid import uuid4

try:
    # audioop is removed from the stdlib in Python 3.13+; audioop-lts (only
    # installable there - see requirements.txt's environment marker) is the
    # official drop-in replacement package for exactly this removal.
    import audioop
except ImportError:
    import audioop_lts as audioop

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .asr import IndicConformerAsr
from .audio_transcode import resample_pcm
from .barge_in import ActiveSpeech
from .clause_chunker import ClauseChunker
from .conversation import ConversationManager
from .llm import LlmClient
from .persistence import CallEventStore
from .settings import AudioSettings, SUPPORTED_LANGUAGES, SecuritySettings
from .tasks import shutdown_worker
from .tts import SvaraTts
from .vad import TenVadSegmenter, VadUpdate

logger = logging.getLogger("aica.telephony")

router = APIRouter()

# G.711 u-law, the standard SIP/PSTN telephony codec - see module docstring
# and BACKEND_COMPLETION.md Sec3.3. This is a transport constant, not a tunable,
# so it lives here rather than in settings.py.
TELEPHONY_SAMPLE_RATE = 8_000
TELEPHONY_ENCODING = "audio/x-mulaw"


def _split_reply_into_clauses(text: str) -> list[str]:
    """One-shot helper: feed a whole LLM reply through ClauseChunker at once.

    Duplicated from main.py's private helper of the same name rather than
    imported: main.py will import this module's `router` once wired in
    (BACKEND_COMPLETION.md Sec3.3), so importing main.py from here would
    create a circular import. The function is a few lines of pure text
    logic with no state, so duplicating it is cheaper than restructuring
    main.py to expose a shared helper module - see the "do not touch
    main.py" scope note this module was built under.
    """
    chunker = ClauseChunker()
    clauses = chunker.feed(text)
    remainder = chunker.flush()
    if remainder:
        clauses.append(remainder)
    return clauses


def _validate_start_event(payload: dict[str, object], settings: AudioSettings) -> str:
    """Validate a Twilio-Media-Streams-shaped "start" event's mediaFormat.

    Returns the language to use for this call: `start.customParameters.language`
    if the caller/trunk config supplied one (Twilio TwiML can pass arbitrary
    `<Parameter>`s), else settings.language - Twilio's own protocol has no
    language field, unlike the browser leg's `call_started.language`.
    """
    start_payload = payload.get("start")
    if not isinstance(start_payload, dict):
        raise ValueError("start event missing 'start' object")

    media_format = start_payload.get("mediaFormat")
    if not isinstance(media_format, dict):
        raise ValueError("start.mediaFormat is required")
    if media_format.get("encoding") != TELEPHONY_ENCODING:
        raise ValueError(f"mediaFormat.encoding must be {TELEPHONY_ENCODING}")
    if media_format.get("sampleRate") != TELEPHONY_SAMPLE_RATE:
        raise ValueError(f"mediaFormat.sampleRate must be {TELEPHONY_SAMPLE_RATE}")
    if media_format.get("channels") != 1:
        raise ValueError("mediaFormat.channels must be 1")

    custom_params = start_payload.get("customParameters")
    if not isinstance(custom_params, dict):
        custom_params = {}
    language = str(custom_params.get("language", settings.language))
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported language: {language}")
    return language


@router.websocket("/ws/telephony")
async def capture_telephony_audio(websocket: WebSocket) -> None:
    """Process one PSTN call's Twilio-Media-Streams-shaped u-law stream.

    Mirrors backend/main.py's capture_browser_audio() almost line for line -
    same queue/worker architecture, same ConversationManager/TTS/barge-in
    wiring - differing only in wire framing (JSON-only, u-law-in-base64
    instead of raw PCM bytes) and the mandatory 8kHz<->16kHz resampling this
    transport needs that the already-16kHz browser leg does not.
    """
    security: SecuritySettings = websocket.app.state.security
    # Pre-accept check: only rejects here if a query param WAS supplied and
    # is wrong, matching capture_browser_audio()'s reject-before-accept
    # behavior. If no query param is present at all, accept provisionally and
    # fall back to checking the "start" event's customParameters below - see
    # the module docstring's Auth section for why both paths exist.
    query_token = websocket.query_params.get("token")
    if security.ws_auth_token and query_token is not None and query_token != security.ws_auth_token:
        await websocket.close(code=4401)
        return

    await websocket.accept()
    connection_id = str(uuid4())
    settings: AudioSettings = websocket.app.state.settings
    asr: IndicConformerAsr = websocket.app.state.asr
    conversation: ConversationManager = websocket.app.state.conversation
    llm: LlmClient = websocket.app.state.llm
    tts: SvaraTts = websocket.app.state.tts
    call_store: CallEventStore = websocket.app.state.call_store
    asr_semaphore: asyncio.Semaphore = websocket.app.state.asr_semaphore
    tts_semaphore: asyncio.Semaphore = websocket.app.state.tts_semaphore
    send_lock = asyncio.Lock()
    language = settings.language
    started = False
    call_sid = ""
    stream_sid = ""
    chunks_received = 0
    bytes_received = 0
    pcm_buffer = bytearray()
    segmenter: TenVadSegmenter | None = None

    call_store.start_call(connection_id)

    async def record_event(payload: dict[str, object]) -> None:
        """Persist a pipeline event for observability/dashboards.

        Unlike capture_browser_audio()'s send_event(), this does NOT also
        send the event over the WebSocket: Twilio Media Streams' outbound
        protocol only understands "media"/"mark"/"clear" messages, so
        internal telemetry events (vad_start, transcript, agent_clause, ...)
        have nowhere meaningful to go on the wire for a real vendor - they
        are recorded to call_store only, exactly like the browser leg's
        events are, just without the redundant (and, for a real vendor,
        invalid) WS send.
        """
        await asyncio.to_thread(call_store.record, connection_id, payload)

    async def send_media_frame(ulaw_bytes: bytes) -> None:
        """Send one outbound audio frame in Twilio Media Streams' documented shape."""
        message = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": base64.b64encode(ulaw_bytes).decode("ascii")},
        }
        async with send_lock:
            await websocket.send_json(message)

    transcription_queue: asyncio.Queue[tuple[np.ndarray, str, str] | None] = asyncio.Queue()
    conversation_queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
    # Tracks whichever speak() task is currently synthesizing/sending audio, so
    # queue_segment() can cancel it the moment the caller barges in - reused
    # unmodified from backend/barge_in.py, same as the browser leg.
    active_speech = ActiveSpeech(settings.barge_in_speech_frames)

    async def speak(text: str) -> None:
        """Chunk one agent reply into clauses, synthesize each, send as u-law over the wire.

        Same full-reply-then-chunk/cancellable-mid-clause shape as
        capture_browser_audio()'s speak() - see that function's docstring.
        The one difference is the outbound leg's transcoding: TTS produces
        int16 PCM at tts.sample_rate, which must come down to 8kHz u-law
        before it can go back out over the telephony leg.
        """
        clauses = _split_reply_into_clauses(text)
        if not clauses:
            return

        await record_event({"type": "agent_speaking_start", "sample_rate": TELEPHONY_SAMPLE_RATE if tts.ready else None})
        if not tts.ready:
            await record_event(
                {
                    "type": "agent_error",
                    "message": (
                        "TTS model is unavailable. backend/tts.py's load() is a "
                        "placeholder until a real svara-TTS reference is wired in."
                    ),
                }
            )

        try:
            for clause in clauses:
                await record_event({"type": "agent_clause", "text": clause})
                if not tts.ready:
                    continue
                async with tts_semaphore:
                    result = await asyncio.to_thread(tts.synthesize, clause, language)
                if result.samples.size:
                    pcm_8k = resample_pcm(result.samples, tts.sample_rate, TELEPHONY_SAMPLE_RATE)
                    ulaw_bytes = audioop.lin2ulaw(pcm_8k.astype("<i2").tobytes(), 2)
                    with suppress(WebSocketDisconnect, RuntimeError):
                        await send_media_frame(ulaw_bytes)
        except asyncio.CancelledError:
            logger.info("agent speech interrupted by barge-in: %s", connection_id)
            await record_event({"type": "agent_interrupted"})
            return

        await record_event({"type": "agent_speaking_end"})

    async def handle_conversation_turns() -> None:
        while True:
            item = await conversation_queue.get()
            try:
                if item is None:
                    return
                kind, payload = item
                reply_text = payload if kind == "greeting" else await conversation.handle_utterance(connection_id, llm, payload)
                speak_task = asyncio.create_task(speak(reply_text))
                active_speech.set(speak_task)
                try:
                    await speak_task
                finally:
                    active_speech.clear(speak_task)
            except Exception as error:
                logger.exception("agent turn failed for %s", connection_id)
                await record_event({"type": "agent_error", "message": str(error)})
            finally:
                conversation_queue.task_done()

    async def queue_segment(update: VadUpdate) -> None:
        if update.speech_started:
            await record_event(
                {
                    "type": "vad_start",
                    "probability": round(update.probability, 4),
                    "onset_rms": round(update.onset_rms, 1),
                    # Guarded because `segmenter` is built on the "start" frame
                    # and typed Optional until then; media is refused before
                    # that, so in practice it is always set by the time an
                    # onset can fire.
                    "noise_floor": round(segmenter.noise_floor, 1) if segmenter else 0.0,
                }
            )
            logger.info("speech started: %s", connection_id)
        # Same gate as main.py: a phone line has more noise than a headset, not
        # less. See ActiveSpeech.note_speech().
        if active_speech.note_speech(update.speech_frame, update.speech_started):
            logger.info("barge-in: cancelling in-flight agent speech for %s", connection_id)
        if not update.speech_ended:
            return

        await record_event(
            {
                "type": "vad_end",
                "probability": round(update.probability, 4),
                "reason": update.end_reason,
            }
        )
        if update.samples is not None:
            await transcription_queue.put((update.samples, language, update.end_reason or "silence"))
        logger.info("speech ended: %s (%s)", connection_id, update.end_reason)

    async def transcribe_segments() -> None:
        while True:
            item = await transcription_queue.get()
            try:
                if item is None:
                    return
                samples, segment_language, reason = item
                duration_ms = round(len(samples) / settings.sample_rate * 1000)
                if not asr.ready:
                    await record_event(
                        {
                            "type": "asr_error",
                            "message": "ASR model is unavailable. Accept the model terms and set HF_TOKEN, then restart the server.",
                        }
                    )
                    continue

                await record_event({"type": "asr_start", "duration_ms": duration_ms, "language": segment_language})
                async with asr_semaphore:
                    transcript = await asyncio.to_thread(asr.transcribe, samples, segment_language)
                transcript = transcript.strip()
                await record_event(
                    {
                        "type": "transcript",
                        "text": transcript,
                        "language": segment_language,
                        "duration_ms": duration_ms,
                        "endpoint_reason": reason,
                    }
                )
                if transcript:
                    await conversation_queue.put(("utterance", transcript))
                logger.info("transcribed %s ms as %s for %s", duration_ms, segment_language, connection_id)
            except Exception as error:
                logger.exception("ASR failed for %s", connection_id)
                await record_event({"type": "asr_error", "message": str(error)})
            finally:
                transcription_queue.task_done()

    worker = asyncio.create_task(transcribe_segments())
    conversation_worker = asyncio.create_task(handle_conversation_turns())
    logger.info("telephony pipeline connected: %s", connection_id)

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            text_message = message.get("text")
            if text_message is None:
                # Twilio Media Streams never sends binary WS frames - audio
                # always arrives as base64 inside a JSON text frame. Ignore
                # any binary frame defensively rather than assume browser-leg
                # framing.
                continue

            try:
                payload = json.loads(text_message)
            except json.JSONDecodeError as error:
                await record_event({"type": "protocol_error", "message": str(error)})
                continue

            event = payload.get("event")

            if event == "start":
                try:
                    language = _validate_start_event(payload, settings)
                except ValueError as error:
                    await record_event({"type": "protocol_error", "message": str(error)})
                    continue

                start_payload = payload.get("start", {})
                call_sid = str(start_payload.get("callSid", ""))
                stream_sid = str(start_payload.get("streamSid") or payload.get("streamSid", ""))

                # Fallback auth path (see module docstring): only reached when
                # no query-param token was supplied at all - a wrong query
                # param was already rejected before accept() above.
                if security.ws_auth_token and query_token is None:
                    custom_params = start_payload.get("customParameters")
                    start_token = custom_params.get("token") if isinstance(custom_params, dict) else None
                    if start_token != security.ws_auth_token:
                        logger.warning("telephony auth failed (start event): %s", connection_id)
                        await websocket.close(code=4401)
                        return

                try:
                    segmenter = TenVadSegmenter(settings)
                except Exception as error:
                    logger.exception("TEN VAD failed to initialize")
                    await record_event({"type": "pipeline_error", "stage": "vad", "message": str(error)})
                    await websocket.close(code=1011)
                    return

                started = True
                await record_event(
                    {
                        "type": "pipeline_configured",
                        "language": language,
                        "call_sid": call_sid,
                        "stream_sid": stream_sid,
                    }
                )
                if conversation.ready:
                    greeting = conversation.start_call(connection_id, agent_name=conversation.settings.agent_name)
                    await conversation_queue.put(("greeting", greeting))
                else:
                    await record_event(
                        {
                            "type": "agent_error",
                            "message": "Conversation Manager is unavailable (master prompt failed to load).",
                        }
                    )
                logger.info("telephony capture started: %s (%s, callSid=%s)", connection_id, language, call_sid)

            elif event == "media":
                if not started or segmenter is None:
                    await record_event({"type": "protocol_error", "message": "received media before start"})
                    continue

                media_payload = payload.get("media")
                b64_payload = media_payload.get("payload") if isinstance(media_payload, dict) else None
                if not b64_payload:
                    continue

                try:
                    ulaw_bytes = base64.b64decode(b64_payload)
                except ValueError as error:
                    await record_event({"type": "protocol_error", "message": f"invalid base64 media payload: {error}"})
                    continue

                chunks_received += 1
                bytes_received += len(ulaw_bytes)

                # u-law (8kHz, 1 byte/sample) -> linear PCM16 (8kHz) -> resample
                # to 16kHz, the rate VAD/ASR are hard-pinned to (settings.py) -
                # see audio_transcode.py and this module's docstring.
                pcm16_8k_bytes = audioop.ulaw2lin(ulaw_bytes, 2)
                pcm16_8k = np.frombuffer(pcm16_8k_bytes, dtype="<i2")
                pcm16_16k = resample_pcm(pcm16_8k, TELEPHONY_SAMPLE_RATE, settings.sample_rate)
                pcm_buffer.extend(pcm16_16k.astype("<i2").tobytes())

                frame_bytes = settings.vad_hop_size * np.dtype("<i2").itemsize
                while len(pcm_buffer) >= frame_bytes:
                    frame = np.frombuffer(pcm_buffer[:frame_bytes], dtype="<i2").copy()
                    del pcm_buffer[:frame_bytes]
                    await queue_segment(segmenter.process(frame))

                if chunks_received % 100 == 0:
                    logger.info(
                        "telephony capture %s: %d chunks, %d bytes received",
                        connection_id,
                        chunks_received,
                        bytes_received,
                    )

            elif event == "stop":
                if segmenter is not None:
                    final_update = segmenter.flush()
                    if final_update:
                        await queue_segment(final_update)
                logger.info("telephony stream stopped: %s", connection_id)
                # Twilio's "stop" event is the definitive end-of-call signal
                # (unlike the browser leg's optional "call_ended" convenience
                # event) - break here rather than waiting for the subsequent
                # WebSocketDisconnect, so cleanup below runs deterministically
                # even if the provider is slow to actually close the socket.
                break

            else:
                # e.g. Twilio's preliminary {"event": "connected", ...} or a
                # "mark" echo - nothing to do, same as capture_browser_audio()'s
                # default branch for unrecognized event types.
                logger.info("telephony event %s: %s", connection_id, event)
    except WebSocketDisconnect:
        pass
    finally:
        if segmenter is not None:
            final_update = segmenter.flush()
            if final_update:
                with suppress(WebSocketDisconnect, RuntimeError):
                    await queue_segment(final_update)
        # Sentinel first, then a bounded wait rather than an immediate cancel:
        # a worker mid-turn otherwise loses the tail of that turn (its last
        # clause, its agent_speaking_end) from the call log, because those are
        # written by a turn's very last statements. See backend/tasks.py.
        await transcription_queue.put(None)
        await conversation_queue.put(None)
        await shutdown_worker(worker, "telephony transcription worker")
        await shutdown_worker(conversation_worker, "telephony conversation worker")
        conversation.end_call(connection_id)
        call_store.end_call(connection_id)
        logger.info(
            "telephony pipeline disconnected: %s (%d chunks, %d bytes)",
            connection_id,
            chunks_received,
            bytes_received,
        )
