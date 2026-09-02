"""Self-check for the Twilio-Media-Streams-shaped telephony ingress adapter.

Exercises backend/telephony.py's router end to end over FastAPI's TestClient
WebSocket support, with every model-backed dependency replaced by a fake
(no real ASR/LLM/TTS model, no network, no real telephony vendor - see the
module docstring in telephony.py for what "code-complete but not live-tested"
means here). Uses the same fakery patterns as test_conversation.py
(_ScriptedLlm, a real ConversationManager with load() bypassed) and
test_vad.py (stubbing `._vad.process` for a deterministic VAD state machine).
"""

from __future__ import annotations

import asyncio
import base64
import sqlite3
import tempfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import audioop
import numpy as np

from . import telephony as telephony_module
from .conversation import ConversationManager
from .llm import LlmReply, ReplyComplete, TextDelta
from .persistence import CallEventStore
from .settings import (
    AudioSettings,
    ConversationSettings,
    LlmSettings,
    PersistenceSettings,
    SecuritySettings,
)
from .telephony import router
from .tts import SynthesisResult
from .vad import TenVadSegmenter as RealTenVadSegmenter

TEMPLATE = "System prompt for {{agent_name}}."
HOP = 256  # AudioSettings.vad_hop_size, 16kHz samples per VAD frame.


class _ScriptedLlm:
    """Same fake as test_conversation.py: scripted replies, no real model."""

    # conversation.py sizes its history trim against these - see
    # test_conversation.py's fake.
    settings = LlmSettings()

    def __init__(self, replies: list[LlmReply]) -> None:
        self._replies = list(replies)
        self.calls: list[list[dict]] = []

    async def stream(self, messages: list[dict], tools: list[dict] | None = None):
        self.calls.append([dict(m) for m in messages])
        reply = self._replies.pop(0)
        for index in range(0, len(reply.content), 7):
            yield TextDelta(reply.content[index : index + 7])
        yield ReplyComplete(reply)

    async def complete(self, messages: list[dict], tools: list[dict]) -> LlmReply:
        self.calls.append([dict(m) for m in messages])
        return self._replies.pop(0)


class _FakeAsr:
    """Scripted transcript; records what it was called with."""

    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.ready = True
        self.calls: list[tuple[np.ndarray, str]] = []

    def transcribe(self, samples: np.ndarray, language: str) -> str:
        self.calls.append((samples, language))
        return self.transcript


class _FakeTts:
    """ready=True, fixed sample rate, returns a small nonempty int16 waveform."""

    def __init__(self, sample_rate: int = 22_050) -> None:
        self.ready = True
        self.sample_rate = sample_rate
        self.calls: list[tuple[str, str]] = []

    def synthesize(self, text: str, language: str) -> SynthesisResult:
        self.calls.append((text, language))
        samples = (np.sin(np.arange(2_205) / 8.0) * 5_000).astype(np.int16)
        return SynthesisResult(samples=samples, sample_rate=self.sample_rate)


def _make_manager(agent_name: str = "Gayathri") -> ConversationManager:
    manager = ConversationManager(ConversationSettings(agent_name=agent_name))
    # These integration tests inject scripted model turns to exercise the
    # telephony path below the production five-flow controller.
    manager._deterministic_flows = False
    # Stub the builder rather than reading the real prompt files: these tests
    # assert telephony plumbing, not prompt content.
    manager.prompts._core = TEMPLATE
    manager.prompts._playbooks = {}
    return manager


def _expected_greeting_clause_count(agent_name: str = "Gayathri") -> int:
    """The real greeting text/clause-splitting logic, computed once so tests
    don't hardcode an assumption about how many clauses OPENING_LINE splits
    into - see clause_chunker.py / conversation.py's OPENING_LINE."""
    probe = _make_manager(agent_name)
    greeting = probe.start_call("probe-conn", agent_name=agent_name)
    return len(telephony_module._split_reply_into_clauses(greeting))


def _build_app(
    *,
    ws_auth_token: str = "",
    llm_replies: list[LlmReply] | None = None,
    transcript: str = "வணக்கம்",
    vad_flags: list[int] | None = None,
    monkeypatch=None,
) -> tuple[FastAPI, Path]:
    app = FastAPI()
    app.include_router(router)

    settings = AudioSettings()
    app.state.settings = settings
    app.state.asr = _FakeAsr(transcript)
    app.state.asr_semaphore = asyncio.Semaphore(2)
    app.state.conversation = _make_manager()
    app.state.llm = _ScriptedLlm(llm_replies or [])
    app.state.tts = _FakeTts()
    app.state.tts_semaphore = asyncio.Semaphore(2)

    tmp_dir = Path(tempfile.mkdtemp())
    db_path = tmp_dir / "call_events.db"
    call_store = CallEventStore(PersistenceSettings(db_path=db_path))
    call_store.load()
    app.state.call_store = call_store
    app.state.security = SecuritySettings(ws_auth_token=ws_auth_token)

    if vad_flags is not None:
        assert monkeypatch is not None, "vad_flags requires a monkeypatch fixture"
        flags_iter = iter(vad_flags)

        def _fake_segmenter(settings_arg):
            seg = RealTenVadSegmenter(settings_arg)
            seg._vad.process = lambda frame: (1.0, next(flags_iter))
            return seg

        monkeypatch.setattr(telephony_module, "TenVadSegmenter", _fake_segmenter)

    return app, db_path


def _start_event(custom_parameters: dict | None = None) -> dict:
    start: dict[str, object] = {
        "callSid": "CA123",
        "streamSid": "MZ123",
        "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8_000, "channels": 1},
    }
    if custom_parameters is not None:
        start["customParameters"] = custom_parameters
    return {"event": "start", "start": start}


def _media_event(ulaw_bytes: bytes) -> dict:
    return {"event": "media", "media": {"payload": base64.b64encode(ulaw_bytes).decode("ascii")}}


def _ulaw_bytes_for_hops(hop_count: int) -> bytes:
    """u-law byte count that yields exactly `hop_count` 16kHz VAD frames after
    ulaw2lin + 8kHz->16kHz resampling (see telephony.py's media-event path):
    1 u-law byte == 1 linear PCM sample @8kHz; resampling to 16kHz doubles the
    sample count; each VAD hop is HOP samples @16kHz."""
    samples_8k = hop_count * HOP // 2
    return bytes(samples_8k)  # content is irrelevant - VAD is stubbed below.


def _last_connection_id(db_path: Path) -> str:
    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute("SELECT connection_id FROM calls ORDER BY started_at DESC LIMIT 1").fetchone()
    finally:
        connection.close()
    assert row is not None, "no call was ever started"
    return row[0]


def _call_is_ended(db_path: Path, connection_id: str) -> bool:
    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(
            "SELECT ended_at FROM calls WHERE connection_id = ?", (connection_id,)
        ).fetchone()
    finally:
        connection.close()
    return row is not None and row[0] is not None


def test_start_event_triggers_greeting_audio() -> None:
    app, db_path = _build_app()
    expected_clauses = _expected_greeting_clause_count()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/telephony") as ws:
            ws.send_json(_start_event())

            for _ in range(expected_clauses):
                outbound = ws.receive_json()
                assert outbound["event"] == "media"
                assert outbound["streamSid"] == "MZ123"
                ulaw_bytes = base64.b64decode(outbound["media"]["payload"])
                pcm = audioop.ulaw2lin(ulaw_bytes, 2)
                assert len(pcm) > 0

    connection_id = _last_connection_id(db_path)
    events = CallEventStore(PersistenceSettings(db_path=db_path)).events_for_call(connection_id)
    event_types = [event["type"] for event in events]
    assert "pipeline_configured" in event_types
    assert event_types.count("agent_speaking_start") == 1
    assert event_types.count("agent_speaking_end") == 1


def test_media_stream_drives_asr_conversation_tts_and_stop_cleans_up(monkeypatch) -> None:
    reply_text = "சரி appointment book பண்ணறேன்"
    transcript_text = "எனக்கு Cardiology-ல ஒரு appointment வேணும்"
    # Enough scripted speech frames to clear the onset debounce, then enough
    # trailing silence to close the turn.
    # Silence tail read from settings, not hard-coded: the segmenter closes
    # the turn after endpoint_silence_frames of silence, so a literal 30
    # here silently stops matching the moment that value is tuned.
    speech_frames = AudioSettings().vad_start_frames
    flags = [1] * speech_frames + [0] * AudioSettings().endpoint_silence_frames

    app, db_path = _build_app(
        llm_replies=[LlmReply(content=reply_text)],
        transcript=transcript_text,
        vad_flags=flags,
        monkeypatch=monkeypatch,
    )
    expected_greeting_clauses = _expected_greeting_clause_count()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/telephony") as ws:
            ws.send_json(_start_event())
            for _ in range(expected_greeting_clauses):
                greeting_frame = ws.receive_json()
                assert greeting_frame["event"] == "media"

            ws.send_json(_media_event(_ulaw_bytes_for_hops(len(flags))))

            reply_frame = ws.receive_json()
            assert reply_frame["event"] == "media"
            assert reply_frame["streamSid"] == "MZ123"
            ulaw_bytes = base64.b64decode(reply_frame["media"]["payload"])
            assert len(audioop.ulaw2lin(ulaw_bytes, 2)) > 0

            ws.send_json({"event": "stop", "stop": {"callSid": "CA123"}})

    asr: _FakeAsr = app.state.asr
    assert len(asr.calls) == 1
    samples, language = asr.calls[0]
    assert language == "ta"
    assert len(samples) == len(flags) * HOP

    llm: _ScriptedLlm = app.state.llm
    assert len(llm.calls) == 1
    assert any(m.get("content") == transcript_text for m in llm.calls[0])

    connection_id = _last_connection_id(db_path)
    events = CallEventStore(PersistenceSettings(db_path=db_path)).events_for_call(connection_id)
    event_types = [event["type"] for event in events]
    assert "vad_start" in event_types
    assert "vad_end" in event_types
    transcripts = [e["text"] for e in events if e["type"] == "transcript"]
    assert transcripts == [transcript_text]

    assert _call_is_ended(db_path, connection_id)


def test_auth_rejects_wrong_query_token_before_accept() -> None:
    app, _ = _build_app(ws_auth_token="secret")

    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/telephony?token=wrong"):
                pass


def test_auth_accepts_correct_query_token() -> None:
    app, _ = _build_app(ws_auth_token="secret")
    expected_clauses = _expected_greeting_clause_count()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/telephony?token=secret") as ws:
            ws.send_json(_start_event())
            for _ in range(expected_clauses):
                outbound = ws.receive_json()
                assert outbound["event"] == "media"


def test_auth_falls_back_to_start_event_custom_parameters_when_no_query_param() -> None:
    app, _ = _build_app(ws_auth_token="secret")
    expected_clauses = _expected_greeting_clause_count()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/telephony") as ws:
            ws.send_json(_start_event(custom_parameters={"token": "secret"}))
            for _ in range(expected_clauses):
                outbound = ws.receive_json()
                assert outbound["event"] == "media"


def test_auth_rejects_missing_custom_parameters_token_when_no_query_param() -> None:
    app, _ = _build_app(ws_auth_token="secret")

    with TestClient(app) as client:
        with client.websocket_connect("/ws/telephony") as ws:
            ws.send_json(_start_event())
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()


def test_stop_before_start_does_not_raise() -> None:
    """Defensive: a malformed/edge-case stream (stop with no prior start)
    must not crash the handler - segmenter is None, cleanup must no-op."""
    app, db_path = _build_app()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/telephony") as ws:
            ws.send_json({"event": "stop", "stop": {"callSid": "CA123"}})

    connection_id = _last_connection_id(db_path)
    assert _call_is_ended(db_path, connection_id)


if __name__ == "__main__":
    import inspect
    import sys

    failures = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and inspect.isfunction(func):
            params = inspect.signature(func).parameters
            try:
                if "monkeypatch" in params:
                    print(f"skipping {name} (needs pytest monkeypatch fixture, run under pytest)")
                    continue
                func()
                print(f"{name}: ok")
            except Exception:
                failures += 1
                import traceback

                traceback.print_exc()
    sys.exit(1 if failures else 0)
