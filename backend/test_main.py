"""Self-check for the /ws/audio handler's orchestration: ready/greeting flow,
call_started validation, auth gate, and the VAD -> ASR -> Conversation ->
TTS turn end to end.

Known gap noted in BACKEND_COMPLETION.md's progress log ("main.py has no
unit tests at all today"). Follows the same fakes-over-real-models pattern as
test_asr.py/test_vad.py/test_conversation.py: app.state is populated by hand
(TestClient never enters the `with` form, so lifespan()'s real model loading
never runs - see below), and the real TenVad native model is swapped for a
scripted stub the same way test_vad.py stubs `segmenter._vad.process`,
except here the stub has to sit one level up (`backend.vad.TenVad` itself),
because the WS handler constructs its own TenVadSegmenter internally with no
injection point.

Deliberately NOT covered here: a barge-in integration test through this WS
surface. ActiveSpeech already has full, deterministic unit coverage
(test_barge_in.py); reproducing that through a real asyncio.to_thread TTS
delay and a second concurrent WS send would be timing-sensitive and flaky in
a way that would make this suite less trustworthy, not more - a known,
deliberate scope cut rather than an oversight.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from . import vad as vad_module
from .conversation import ConversationManager
from .llm import LlmReply, ReplyComplete, TextDelta
from .main import app
from .persistence import CallEventStore
from .settings import (
    AudioSettings,
    ConversationSettings,
    LlmSettings,
    PersistenceSettings,
    SecuritySettings,
)

PROMPT_TEMPLATE = "Hello {{agent_name}}."


class _FakeAsr:
    def __init__(self, transcript: str = "", partial_transcript: str = "", ready: bool = True) -> None:
        self._transcript = transcript
        self._partial_transcript = partial_transcript
        self.ready = ready
        self.calls: list[tuple] = []
        self.partial_calls: list[tuple] = []

    def transcribe(self, samples, language: str) -> str:
        self.calls.append((samples, language))
        return self._transcript

    def transcribe_partial(self, samples, language: str) -> str:
        self.partial_calls.append((samples, language))
        return self._partial_transcript


class _FakeTts:
    def __init__(self, ready: bool = True) -> None:
        self.ready = ready
        self.sample_rate = 22_050
        self.calls: list[str] = []

    def synthesize(self, text: str, language: str):
        self.calls.append(text)
        from .tts import SynthesisResult

        return SynthesisResult(samples=np.array([1, 2, 3], dtype=np.int16), sample_rate=self.sample_rate)


def test_tts_warmup_covers_every_scripted_conversation_line() -> None:
    """New guard/recovery lines must keep the startup latency optimization."""
    from types import SimpleNamespace

    from .conversation import (
        _CANNOT_RECALL,
        _ECHO_RECOVERY,
        _EMERGENCY_STUCK_REPLIES,
        _GO_AHEAD,
        _OUT_OF_SCOPE,
        _STUCK_REPLIES,
        split_reply_into_clauses,
    )
    from .main import _warm

    class _WarmTts:
        settings = SimpleNamespace(language="ta")

        def __init__(self) -> None:
            self.calls: list[str] = []

        def synthesize(self, text: str, language: str) -> None:
            assert language == "ta"
            self.calls.append(text)

    tts = _WarmTts()
    _warm(tts)

    expected = {
        _GO_AHEAD,
        _CANNOT_RECALL,
        _ECHO_RECOVERY,
        *_STUCK_REPLIES,
        *_EMERGENCY_STUCK_REPLIES,
        *split_reply_into_clauses(_OUT_OF_SCOPE),
    }
    assert expected <= set(tts.calls)


class _ScriptedLlm:
    """Same pattern as test_conversation.py's fake - scripted replies, no network."""

    ready = True
    # See test_conversation.py's fake: conversation.py sizes its history
    # trim against these.
    settings = LlmSettings()

    def __init__(self, replies: list[LlmReply]) -> None:
        self._replies = list(replies)
        # Real caller turns only - a prewarm is not the model being asked
        # anything, and counting it would hide a turn that never happened.
        self.turns: list[list[dict]] = []

    async def stream(self, messages, tools=None, max_tokens=None):
        # ConversationManager.prewarm() asks for a single token purely to make
        # the server evaluate the prompt into its cache, and throws the output
        # away. Modelled here so a prewarm does not silently eat the reply the
        # test scripted for the caller's actual turn.
        if max_tokens == 1:
            yield ReplyComplete(LlmReply(content=""))
            return
        self.turns.append([dict(m) for m in messages])
        reply = self._replies.pop(0)
        for index in range(0, len(reply.content), 7):
            yield TextDelta(reply.content[index : index + 7])
        yield ReplyComplete(reply)

    async def complete(self, messages, tools) -> LlmReply:
        async for event in self.stream(messages, tools):
            if isinstance(event, ReplyComplete):
                return event.reply
        raise AssertionError("scripted stream produced no ReplyComplete")


class _FakeTenVad:
    """Replaces the native ten_vad model with a scripted flag sequence.

    TenVadSegmenter is constructed fresh inside the WS handler with no
    injection point (see module docstring), so the fake has to sit at the
    import site: `backend.vad.TenVad` itself, not an instance.
    """

    def __init__(self, hop_size: int, threshold: float) -> None:
        self._flags = iter(_CURRENT_FLAGS)

    def process(self, frame):
        flag = next(self._flags, 0)
        return (1.0, flag)


_CURRENT_FLAGS: list[int] = []


@contextlib.contextmanager
def _scripted_vad(flags: list[int]):
    global _CURRENT_FLAGS
    _CURRENT_FLAGS = flags
    original = vad_module.TenVad
    vad_module.TenVad = _FakeTenVad
    try:
        yield
    finally:
        vad_module.TenVad = original


def _make_conversation() -> ConversationManager:
    manager = ConversationManager(ConversationSettings())
    # Stub the builder rather than reading the real prompt files: these tests
    # assert websocket/session plumbing, not prompt content.
    manager.prompts._core = PROMPT_TEMPLATE
    manager.prompts._playbooks = {}
    return manager


def _make_call_store() -> CallEventStore:
    store = CallEventStore(PersistenceSettings(db_path=Path(tempfile.mkdtemp()) / "call_events.db"))
    store.load()
    return store


def _set_app_state(
    *,
    asr=None,
    tts=None,
    conversation=None,
    llm=None,
    security: SecuritySettings | None = None,
) -> None:
    import asyncio

    app.state.settings = AudioSettings()
    app.state.asr = asr if asr is not None else _FakeAsr()
    app.state.asr_semaphore = asyncio.Semaphore(2)
    app.state.conversation = conversation if conversation is not None else _make_conversation()
    app.state.llm = llm if llm is not None else _ScriptedLlm([])
    app.state.tts = tts if tts is not None else _FakeTts()
    app.state.tts_semaphore = asyncio.Semaphore(2)
    app.state.call_store = _make_call_store()
    app.state.security = security if security is not None else SecuritySettings()


def _next_json(ws) -> dict:
    """Read messages until a JSON text frame arrives, skipping binary audio frames."""
    while True:
        message = ws.receive()
        if message.get("type") == "websocket.send" and message.get("text") is not None:
            return json.loads(message["text"])
        if "text" in message and message["text"] is not None:
            return json.loads(message["text"])
        # else: a binary audio frame - skip it, the tests below assert on
        # events, not on synthesized audio bytes.


_VALID_CALL_STARTED = {
    "type": "call_started",
    "audio_format": "pcm_s16le",
    "sample_rate": 16_000,
    "channels": 1,
    "language": "ta",
}


# Loudness gates turn ONSET (AudioSettings.vad_onset_min_rms), so scripted VAD
# flags are no longer enough on their own - the PCM behind a "speech" frame has
# to actually be at a speaking level. Zero-filled audio now (correctly) opens
# no turn at all, which is the point of the gate.
_SPOKEN_SAMPLE = 2600
_ROOM_SAMPLE = 40


def _audio_for(flags: list[int]) -> bytes:
    """PCM matching the scripted flags: speaking level where the VAD says speech."""
    import numpy as np

    hop = AudioSettings().vad_hop_size
    frames = [
        np.full(hop, _SPOKEN_SAMPLE if flag else _ROOM_SAMPLE, dtype="<i2") for flag in flags
    ]
    return np.concatenate(frames).tobytes()


def _drain_agent_turn(ws) -> list[dict]:
    """Read agent_speaking_start ... agent_speaking_end, returning every event in between."""
    events = [_next_json(ws)]
    assert events[0]["type"] == "agent_speaking_start"
    while events[-1]["type"] != "agent_speaking_end":
        events.append(_next_json(ws))
    return events


def test_ready_event_reports_component_status() -> None:
    _set_app_state(asr=_FakeAsr(ready=False), tts=_FakeTts(ready=False))
    client = TestClient(app)
    with client.websocket_connect("/ws/audio") as ws:
        ready = _next_json(ws)

    assert ready["type"] == "ready"
    assert ready["asr_ready"] is False
    assert ready["tts_ready"] is False
    assert ready["conversation_ready"] is True


def test_call_started_with_wrong_sample_rate_is_a_protocol_error() -> None:
    _set_app_state()
    client = TestClient(app)
    with client.websocket_connect("/ws/audio") as ws:
        _next_json(ws)  # ready
        ws.send_json({**_VALID_CALL_STARTED, "sample_rate": 8_000})
        error = _next_json(ws)

    assert error["type"] == "protocol_error"
    assert "sample_rate" in error["message"]


def test_audio_before_call_started_is_a_protocol_error() -> None:
    _set_app_state()
    client = TestClient(app)
    with client.websocket_connect("/ws/audio") as ws:
        _next_json(ws)  # ready
        ws.send_bytes(b"\x00" * 512)
        error = _next_json(ws)

    assert error["type"] == "protocol_error"
    assert "call_started" in error["message"]


def test_valid_call_started_configures_pipeline_and_speaks_greeting() -> None:
    tts = _FakeTts(ready=True)
    _set_app_state(tts=tts)
    client = TestClient(app)
    with client.websocket_connect("/ws/audio") as ws:
        _next_json(ws)  # ready
        ws.send_json(_VALID_CALL_STARTED)
        configured = _next_json(ws)
        turn = _drain_agent_turn(ws)

    assert configured == {"type": "pipeline_configured", "language": "ta"}
    clause_texts = [e["text"] for e in turn if e["type"] == "agent_clause"]
    assert clause_texts, "greeting should produce at least one spoken clause"
    # The greeting is conversation.py's hardcoded OPENING_LINE (see Sec5A),
    # not this test's PROMPT_TEMPLATE - just confirm the real per-call
    # substitution (agent_name) made it into what got spoken.
    assert "Gayathri" in "".join(clause_texts)
    assert tts.calls, "TTS should have been invoked for the greeting"


def test_a_second_call_started_is_refused_rather_than_restarting_the_call() -> None:
    """One call per socket.

    A second call_started used to run the whole opening again on the same
    connection: a fresh CallSession (throwing away the history the call had
    built), the greeting queued and spoken a second time, and a second prewarm
    task assigned over the first - which orphaned it, since teardown cancels
    only the task the variable still points at.
    """
    tts = _FakeTts(ready=True)
    _set_app_state(tts=tts)
    client = TestClient(app)
    with client.websocket_connect("/ws/audio") as ws:
        _next_json(ws)  # ready
        ws.send_json(_VALID_CALL_STARTED)
        _next_json(ws)  # pipeline_configured
        first_turn = _drain_agent_turn(ws)

        ws.send_json(_VALID_CALL_STARTED)
        error = _next_json(ws)

    assert error["type"] == "protocol_error"
    assert "already" in error["message"]
    # ...and no second greeting followed it.
    assert [e["type"] for e in first_turn].count("agent_speaking_end") == 1


def test_user_text_before_call_started_is_a_protocol_error() -> None:
    _set_app_state()
    client = TestClient(app)
    with client.websocket_connect("/ws/audio") as ws:
        _next_json(ws)  # ready
        ws.send_json({"type": "user_text", "text": "hello"})
        error = _next_json(ws)

    assert error["type"] == "protocol_error"
    assert "user_text" in error["message"]


def test_user_text_drives_a_conversation_turn_without_audio() -> None:
    """Typed caller input (DirectTestingPanel) bypasses VAD/ASR entirely but
    feeds the same conversation_queue -> LLM -> TTS path a spoken turn does."""
    tts = _FakeTts(ready=True)
    llm = _ScriptedLlm([LlmReply(content="Cardiology-ல appointment வேணும், சரியா?")])
    _set_app_state(tts=tts, llm=llm)
    client = TestClient(app)

    with client.websocket_connect("/ws/audio") as ws:
        _next_json(ws)  # ready
        ws.send_json(_VALID_CALL_STARTED)
        _next_json(ws)  # pipeline_configured
        _drain_agent_turn(ws)  # greeting - scripted before any LLM call is made

        ws.send_json({"type": "user_text", "text": "Cardiology appointment வேணும்"})
        turn = _drain_agent_turn(ws)

    clause_texts = [e["text"] for e in turn if e["type"] == "agent_clause"]
    assert "".join(clause_texts)
    assert tts.calls, "the LLM's reply to typed input should have been synthesized"


class _OverlapTrackingTts:
    """Records the high-water mark of concurrent synthesize() calls.

    synthesize() is dispatched through asyncio.to_thread, so these run on
    worker threads and the counter needs a lock. The sleep is what gives a
    second clause the chance to start before the first has finished - without
    it every call would complete too fast for an overlap to be observable
    either way, and the test would pass whether or not the pipeline works.
    """

    def __init__(self) -> None:
        self.ready = True
        self.sample_rate = 22_050
        self.calls: list[str] = []
        self.max_concurrent = 0
        self._active = 0
        self._lock = threading.Lock()

    def synthesize(self, text: str, language: str):
        with self._lock:
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
            self.calls.append(text)
        time.sleep(0.15)
        with self._lock:
            self._active -= 1
        from .tts import SynthesisResult

        return SynthesisResult(samples=np.array([1, 2, 3], dtype=np.int16), sample_rate=self.sample_rate)


def test_clauses_are_synthesized_concurrently_not_one_after_another() -> None:
    """The gap the caller hears between sentences is a whole TTS round-trip.

    Each clause used to be synthesized only after the previous one had been
    sent, so a three-sentence reply cost three sequential round-trips and the
    default engine is a network call (849ms-10.8s measured). Overlapping them
    is what makes the agent's speech continuous, and nothing else in the suite
    would notice if that overlap were lost - the audio still arrives, just
    seconds late, which no assertion on message CONTENT can see.
    """
    tts = _OverlapTrackingTts()
    llm = _ScriptedLlm([LlmReply(content="ஒன்று சொல்றேன். ரெண்டு சொல்றேன். மூணு சொல்றேன்.")])
    _set_app_state(tts=tts, llm=llm)
    client = TestClient(app)

    with client.websocket_connect("/ws/audio") as ws:
        _next_json(ws)  # ready
        ws.send_json(_VALID_CALL_STARTED)
        _next_json(ws)  # pipeline_configured
        _drain_agent_turn(ws)  # greeting

        ws.send_json({"type": "user_text", "text": "சொல்லுங்க"})
        _drain_agent_turn(ws)

    assert len(tts.calls) >= 2, f"expected a multi-clause reply, synthesized {tts.calls}"
    assert tts.max_concurrent > 1, (
        "clauses were synthesized strictly one after another - the caller hears "
        "one full TTS round-trip of silence between every sentence"
    )


def test_blank_user_text_does_not_trigger_a_conversation_turn() -> None:
    llm = _ScriptedLlm([LlmReply(content="only reply")])
    _set_app_state(llm=llm)
    client = TestClient(app)

    with client.websocket_connect("/ws/audio") as ws:
        _next_json(ws)  # ready
        ws.send_json(_VALID_CALL_STARTED)
        _next_json(ws)  # pipeline_configured
        _drain_agent_turn(ws)  # greeting - the OPENING_LINE, not an LLM call

        # If blank text incorrectly queued a turn, it would consume the one
        # scripted reply here, and the real question below would then pop
        # from an empty list - surfacing as an agent_error instead of a
        # normal agent_speaking_start/agent_clause turn.
        ws.send_json({"type": "user_text", "text": "   "})
        ws.send_json({"type": "user_text", "text": "real question"})
        turn = _drain_agent_turn(ws)

    clause_texts = [e["text"] for e in turn if e["type"] == "agent_clause"]
    assert "only reply" in "".join(clause_texts)


def test_conversation_not_ready_reports_agent_error_on_call_started() -> None:
    manager = ConversationManager(ConversationSettings())  # load() never called - not ready
    _set_app_state(conversation=manager)
    client = TestClient(app)
    with client.websocket_connect("/ws/audio") as ws:
        _next_json(ws)  # ready
        ws.send_json(_VALID_CALL_STARTED)
        configured = _next_json(ws)
        error = _next_json(ws)

    assert configured == {"type": "pipeline_configured", "language": "ta"}
    assert error["type"] == "agent_error"
    assert "Conversation Manager" in error["message"]


def test_auth_rejects_connection_with_wrong_or_missing_token() -> None:
    _set_app_state(security=SecuritySettings(ws_auth_token="secret-token"))
    client = TestClient(app)
    try:
        with client.websocket_connect("/ws/audio"):
            raise AssertionError("connection without a token must be rejected")
    except Exception:
        pass  # server closes before accept() - the client sees a handshake failure


def test_auth_accepts_connection_with_correct_token() -> None:
    _set_app_state(security=SecuritySettings(ws_auth_token="secret-token"))
    client = TestClient(app)
    with client.websocket_connect("/ws/audio?token=secret-token") as ws:
        ready = _next_json(ws)

    assert ready["type"] == "ready"


def test_speech_turn_flows_through_vad_asr_conversation_tts() -> None:
    asr = _FakeAsr(transcript="Cardiology appointment வேணும்")
    tts = _FakeTts(ready=True)
    llm = _ScriptedLlm([LlmReply(content="Cardiology-ல appointment வேணும், சரியா?")])
    _set_app_state(asr=asr, tts=tts, llm=llm)
    client = TestClient(app)

    # Enough speech frames to clear the onset debounce (vad_start_frames), then
    # enough silence to close the turn. Both read from settings rather than
    # hard-coded, so tuning either knob cannot silently stop exercising a turn.
    _s = AudioSettings()
    flags = [1] * _s.vad_start_frames + [0] * (_s.endpoint_silence_frames + 1)
    audio = _audio_for(flags)

    with _scripted_vad(flags):
        with client.websocket_connect("/ws/audio") as ws:
            _next_json(ws)  # ready
            ws.send_json(_VALID_CALL_STARTED)
            _next_json(ws)  # pipeline_configured
            _drain_agent_turn(ws)  # greeting

            ws.send_bytes(audio)

            vad_start = _next_json(ws)
            vad_end = _next_json(ws)
            asr_start = _next_json(ws)
            transcript = _next_json(ws)
            turn = _drain_agent_turn(ws)

    assert vad_start["type"] == "vad_start"
    assert vad_end == {"type": "vad_end", "probability": vad_end["probability"], "reason": "silence"}
    assert asr_start["type"] == "asr_start"
    assert transcript["type"] == "transcript"
    assert transcript["text"] == "Cardiology appointment வேணும்"
    assert asr.calls, "ASR should have been invoked on the segmented utterance"
    clause_texts = [e["text"] for e in turn if e["type"] == "agent_clause"]
    assert "".join(clause_texts)
    assert tts.calls, "the LLM's reply should have been synthesized"


def test_long_utterance_emits_partial_transcripts_before_the_turn_ends() -> None:
    asr = _FakeAsr(transcript="final text", partial_transcript="interim text")
    llm = _ScriptedLlm([LlmReply(content="acknowledged")])
    _set_app_state(asr=asr, llm=llm)
    client = TestClient(app)

    # PARTIAL_TRANSCRIPT_INTERVAL_FRAMES is 30 - 65 speech frames guarantees
    # at least one interim decode fires while still mid-utterance, before the
    # trailing silence closes the turn.
    flags = [1] * 65 + [0] * (AudioSettings().endpoint_silence_frames + 1)
    audio = _audio_for(flags)

    with _scripted_vad(flags):
        with client.websocket_connect("/ws/audio") as ws:
            _next_json(ws)  # ready
            ws.send_json(_VALID_CALL_STARTED)
            _next_json(ws)  # pipeline_configured
            _drain_agent_turn(ws)  # greeting

            ws.send_bytes(audio)

            event = _next_json(ws)
            assert event["type"] == "vad_start"
            partials = []
            while event["type"] != "vad_end":
                event = _next_json(ws)
                if event["type"] == "partial_transcript":
                    partials.append(event)
            _next_json(ws)  # asr_start
            transcript = _next_json(ws)
            _drain_agent_turn(ws)

    assert partials, "a 65-frame utterance should have produced at least one partial_transcript"
    assert all(p["text"] == "interim text" for p in partials)
    assert asr.partial_calls, "transcribe_partial should have been invoked while still mid-utterance"
    assert transcript["type"] == "transcript" and transcript["text"] == "final text"


def test_the_agent_hearing_itself_does_not_become_a_caller_turn() -> None:
    """Mic left open, speakers on: the agent's own sentence leaks back in.

    Unlike background noise this transcribes to REAL WORDS, so the
    empty-transcript check cannot catch it - and acting on it means the agent
    answering its own question. The greeting is what has just been said, so a
    transcript made of the greeting's own words is an echo by construction.
    """
    echo = "வணக்கம் அருவி ஹாஸ்பிட்டல் உங்களுக்கு எப்படி help பண்ணலாம்"
    asr = _FakeAsr(transcript=echo)
    llm = _ScriptedLlm([LlmReply(content="இது ஒருபோதும் பேசப்படக்கூடாது")])
    _set_app_state(asr=asr, llm=llm)
    client = TestClient(app)

    _s = AudioSettings()
    flags = [1] * _s.vad_start_frames + [0] * (_s.endpoint_silence_frames + 1)
    audio = _audio_for(flags)

    with _scripted_vad(flags):
        with client.websocket_connect("/ws/audio") as ws:
            _next_json(ws)                      # ready
            ws.send_json(_VALID_CALL_STARTED)
            _next_json(ws)                      # pipeline_configured
            _drain_agent_turn(ws)               # greeting

            ws.send_bytes(audio)
            _next_json(ws)                      # vad_start
            event = _next_json(ws)
            while event["type"] not in {"echo_discarded", "transcript"}:
                event = _next_json(ws)

    assert event["type"] == "echo_discarded", (
        f"the agent's own words came back as a {event['type']} event"
    )
    assert event["text"] == echo
    # The decisive one: nothing was sent to the model, so the agent cannot
    # have answered itself.
    assert llm.turns == [], "the agent started a turn in reply to its own voice"


def test_a_real_caller_turn_is_still_heard_while_the_agent_is_speaking() -> None:
    """The echo guard must not deafen the agent to an actual caller."""
    asr = _FakeAsr(transcript="எனக்கு Cardiology-ல appointment book பண்ணணும்")
    llm = _ScriptedLlm([LlmReply(content="கண்டிப்பா சார்.")])
    _set_app_state(asr=asr, llm=llm)
    client = TestClient(app)

    _s = AudioSettings()
    flags = [1] * _s.vad_start_frames + [0] * (_s.endpoint_silence_frames + 1)
    audio = _audio_for(flags)

    with _scripted_vad(flags):
        with client.websocket_connect("/ws/audio") as ws:
            _next_json(ws)
            ws.send_json(_VALID_CALL_STARTED)
            _next_json(ws)
            _drain_agent_turn(ws)

            ws.send_bytes(audio)
            event = _next_json(ws)
            while event["type"] not in {"echo_discarded", "transcript"}:
                event = _next_json(ws)
            assert event["type"] == "transcript", "a real caller turn was thrown away as echo"
            _drain_agent_turn(ws)

    assert llm.turns, "the caller's turn never reached the model"


def test_asr_not_ready_reports_asr_error_instead_of_transcript() -> None:
    asr = _FakeAsr(ready=False)
    _set_app_state(asr=asr)
    client = TestClient(app)

    _s = AudioSettings()
    flags = [1] * _s.vad_start_frames + [0] * (_s.endpoint_silence_frames + 1)
    audio = _audio_for(flags)

    with _scripted_vad(flags):
        with client.websocket_connect("/ws/audio") as ws:
            _next_json(ws)  # ready
            ws.send_json(_VALID_CALL_STARTED)
            _next_json(ws)  # pipeline_configured
            _drain_agent_turn(ws)  # greeting

            ws.send_bytes(audio)

            _next_json(ws)  # vad_start
            _next_json(ws)  # vad_end
            error = _next_json(ws)

    assert error["type"] == "asr_error"
    assert "unavailable" in error["message"]


if __name__ == "__main__":
    test_ready_event_reports_component_status()
    test_call_started_with_wrong_sample_rate_is_a_protocol_error()
    test_audio_before_call_started_is_a_protocol_error()
    test_valid_call_started_configures_pipeline_and_speaks_greeting()
    test_user_text_before_call_started_is_a_protocol_error()
    test_user_text_drives_a_conversation_turn_without_audio()
    test_blank_user_text_does_not_trigger_a_conversation_turn()
    test_conversation_not_ready_reports_agent_error_on_call_started()
    test_auth_rejects_connection_with_wrong_or_missing_token()
    test_auth_accepts_connection_with_correct_token()
    test_speech_turn_flows_through_vad_asr_conversation_tts()
    test_long_utterance_emits_partial_transcripts_before_the_turn_ends()
    test_the_agent_hearing_itself_does_not_become_a_caller_turn()
    test_a_real_caller_turn_is_still_heard_while_the_agent_is_speaking()
    test_asr_not_ready_reports_asr_error_instead_of_transcript()
    print("ok")
