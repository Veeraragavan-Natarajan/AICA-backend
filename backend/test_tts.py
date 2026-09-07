"""Self-check for SvaraTts's synthesis interface (load()/ready/synthesize()).

Stubs the model directly the way test_asr.py bypasses IndicConformerAsr.load()
- the real svara-TTS import is a placeholder, see backend/tts.py.
"""

from __future__ import annotations

import numpy as np

from .settings import TtsSettings
from .tts import SvaraTts


class _FakeModel:
    """Records how synthesis was called; returns a scripted float32 waveform."""

    def __init__(self, waveform) -> None:
        self.waveform = waveform
        self.calls: list[dict] = []

    def synthesize(self, text, *, language, voice_reference_path):
        self.calls.append({"text": text, "language": language, "voice_reference_path": voice_reference_path})
        return self.waveform


def _make_tts(waveform, sample_rate: int = 22_050) -> tuple[SvaraTts, _FakeModel]:
    tts = SvaraTts(TtsSettings(voice_reference_path="voices/agent.wav"))
    model = _FakeModel(waveform)
    tts._model = model
    tts._sample_rate = sample_rate
    return tts, model


def test_empty_text_short_circuits() -> None:
    tts, model = _make_tts([0.5, -0.5])

    result = tts.synthesize("   ", "ta")

    assert result.samples.size == 0
    assert result.sample_rate == 22_050
    assert model.calls == []


def test_unloaded_model_raises() -> None:
    tts = SvaraTts(TtsSettings())
    try:
        tts.synthesize("வணக்கம்", "ta")
    except RuntimeError:
        return
    raise AssertionError("synthesize must refuse to run before load()")


def test_sample_rate_raises_before_load() -> None:
    tts = SvaraTts(TtsSettings())
    try:
        _ = tts.sample_rate
    except RuntimeError:
        return
    raise AssertionError("sample_rate must refuse to be read before load()")


def test_model_is_called_with_text_language_and_voice_reference() -> None:
    tts, model = _make_tts([0.0, 0.5])

    tts.synthesize("  Cardiology appointment  ", "ta")

    call = model.calls[0]
    assert call["text"] == "Cardiology appointment"
    assert call["language"] == "ta"
    assert call["voice_reference_path"] == "voices/agent.wav"


def test_waveform_is_scaled_and_converted_to_int16() -> None:
    tts, _ = _make_tts([0.0, 1.0, -1.0, 0.5])

    result = tts.synthesize("hello", "ta")

    assert result.samples.dtype == np.int16
    assert list(result.samples) == [0, 32767, -32767, 16383]
    assert result.sample_rate == 22_050


def test_out_of_range_waveform_is_clipped_before_scaling() -> None:
    tts, _ = _make_tts([2.0, -2.0])

    result = tts.synthesize("hello", "ta")

    assert list(result.samples) == [32767, -32767]


if __name__ == "__main__":
    test_empty_text_short_circuits()
    test_unloaded_model_raises()
    test_sample_rate_raises_before_load()
    test_model_is_called_with_text_language_and_voice_reference()
    test_waveform_is_scaled_and_converted_to_int16()
    test_out_of_range_waveform_is_clipped_before_scaling()
    print("ok")


# --- engine selection + EdgeTts -------------------------------------------
# EdgeTts is exercised against a stubbed MP3 stream rather than the network:
# the decode/mix/scale path is ours and worth testing, the Microsoft endpoint
# is not, and a unit test must not depend on it being reachable.

import asyncio
import io

import pytest

from .tts import EdgeTts, create_tts


def test_create_tts_selects_the_engine_named_by_settings() -> None:
    assert isinstance(create_tts(TtsSettings(cache_dir="", engine="edge")), EdgeTts)
    assert isinstance(create_tts(TtsSettings(engine="svara")), SvaraTts)


def test_unknown_engine_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        TtsSettings(engine="festival")


def test_edge_defaults_to_the_female_voice_for_the_configured_language() -> None:
    tts = EdgeTts(TtsSettings(cache_dir="", engine="edge", language="ta"))
    tts.load()

    assert tts._voice == "ta-IN-PallaviNeural"
    assert tts.ready


def test_edge_voice_override_wins_over_the_language_default() -> None:
    tts = EdgeTts(TtsSettings(cache_dir="", engine="edge", language="ta", voice="ta-IN-ValluvarNeural"))
    tts.load()

    assert tts._voice == "ta-IN-ValluvarNeural"


def _stub_mp3(monkeypatch, tts: EdgeTts, waveform, sample_rate: int) -> None:
    import soundfile

    buffer = io.BytesIO()
    soundfile.write(buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
    monkeypatch.setattr(tts, "_stream_mp3", lambda text: _immediate(buffer.getvalue()))


async def _immediate(value):
    return value


def _audio_bytes(waveform, sample_rate: int) -> bytes:
    """A decodable body for the stubbed engine. WAV, not MP3: synthesize()
    hands whatever it got straight to soundfile, which sniffs the format, and
    WAV needs no libsndfile MP3 encoder to exist on the test machine."""
    import soundfile

    buffer = io.BytesIO()
    soundfile.write(buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def test_synthesize_returns_int16_pcm_at_the_decoded_rate(monkeypatch) -> None:
    tts = EdgeTts(TtsSettings(cache_dir="", engine="edge", language="ta"))
    tts.load()
    _stub_mp3(monkeypatch, tts, np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32), 24_000)

    result = tts.synthesize("வணக்கம்", "ta")

    assert result.samples.dtype == np.int16
    assert result.sample_rate == 24_000
    assert tts.sample_rate == 24_000


def test_synthesize_downmixes_stereo_to_mono(monkeypatch) -> None:
    tts = EdgeTts(TtsSettings(cache_dir="", engine="edge", language="ta"))
    tts.load()
    stereo = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    _stub_mp3(monkeypatch, tts, stereo, 24_000)

    result = tts.synthesize("வணக்கம்", "ta")

    assert result.samples.ndim == 1
    assert result.samples.shape == (2,)


def test_blank_text_never_reaches_the_network(monkeypatch) -> None:
    tts = EdgeTts(TtsSettings(cache_dir="", engine="edge", language="ta"))
    tts.load()

    def _explode(text):
        raise AssertionError("blank clause must not be synthesized")

    monkeypatch.setattr(tts, "_stream_mp3", _explode)

    assert tts.synthesize("   ", "ta").samples.size == 0


def test_synthesize_before_load_raises() -> None:
    with pytest.raises(RuntimeError):
        EdgeTts(TtsSettings(cache_dir="", engine="edge")).synthesize("வணக்கம்", "ta")


def test_synthesize_works_from_inside_a_running_event_loop(monkeypatch) -> None:
    # synthesize() is sync with an async engine underneath. main.py reaches it
    # via asyncio.to_thread (no loop in that thread), but an async caller -
    # scripts/transcript_log.py, a notebook - calls it directly, where a naive
    # asyncio.run() raises "cannot be called from a running event loop".
    tts = EdgeTts(TtsSettings(cache_dir="", engine="edge", language="ta"))
    tts.load()
    _stub_mp3(monkeypatch, tts, np.array([0.25, -0.25], dtype=np.float32), 24_000)

    async def call_from_loop():
        return tts.synthesize("வணக்கம்", "ta")

    result = asyncio.run(call_from_loop())

    assert result.samples.dtype == np.int16
    assert result.samples.size == 2


def test_a_stalled_endpoint_gives_up_instead_of_hanging_the_turn(monkeypatch) -> None:
    """Measured failure: an unreachable endpoint cost 21s on ONE clause, and the
    sender holds every later clause's audio behind it."""
    import sys
    import time
    import types

    class _Stalling:
        def __init__(self, text, voice, rate=None) -> None:
            pass

        async def stream(self):
            await asyncio.sleep(5)  # a regression fails here in 5s, not forever
            yield {"type": "audio", "data": b""}

    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=_Stalling))
    tts = EdgeTts(TtsSettings(cache_dir="", timeout_seconds=0.2))
    tts._voice = "ta-IN-PallaviNeural"

    started = time.perf_counter()
    try:
        tts.synthesize("வணக்கம்", "ta")
        raise AssertionError("a stalled endpoint must not return audio")
    except RuntimeError as error:
        assert "timed out" in str(error)
    elapsed = time.perf_counter() - started
    # One timeout, not two: retrying a stall just stalls again.
    assert elapsed < 1.0, f"gave up after {elapsed:.2f}s, timeout was 0.2s"


def test_a_repeated_clause_is_not_refetched_so_the_greeting_survives_a_dead_link(monkeypatch) -> None:
    """Every call speaks the same four greeting clauses and the same holding
    line. Re-fetching them is both 0.9s of avoidable latency per call and the
    reason a caller gets no voice at all the moment the endpoint stalls."""
    import sys
    import types

    fetches: list[str] = []

    class _Counting:
        def __init__(self, text, voice, rate=None) -> None:
            fetches.append(text)

        async def stream(self):
            yield {"type": "audio", "data": _audio_bytes(np.array([0.5], dtype=np.float32), 24_000)}

    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=_Counting))
    tts = EdgeTts(TtsSettings(cache_dir="", engine="edge", language="ta"))
    tts._voice = "ta-IN-PallaviNeural"

    first = tts.synthesize("வணக்கம்", "ta")
    second = tts.synthesize("வணக்கம்", "ta")

    assert fetches == ["வணக்கம்"], f"the endpoint was hit {len(fetches)} times for one clause"
    assert second.samples.tolist() == first.samples.tolist()


def test_a_clause_truncated_by_the_timeout_keeps_its_audio_and_is_never_cached(monkeypatch) -> None:
    """A stall part-way through the body used to throw away every byte that had
    already arrived. Keeping them is a clipped word instead of silence - but a
    clipped clause must never be served again from cache."""
    import sys
    import types

    body = _audio_bytes(np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32), 24_000)

    class _StallsMidBody:
        def __init__(self, text, voice, rate=None) -> None:
            pass

        async def stream(self):
            yield {"type": "audio", "data": body}
            await asyncio.sleep(5)  # never delivers the rest

    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=_StallsMidBody))
    tts = EdgeTts(TtsSettings(cache_dir="", timeout_seconds=0.2))
    tts._voice = "ta-IN-PallaviNeural"

    result = tts.synthesize("வணக்கம்", "ta")

    assert result.samples.size > 0, "the bytes that arrived before the deadline were discarded"
    assert tts._mp3_cache == {}, "a truncated clause was cached and will be replayed clipped forever"


def _counting_edge(monkeypatch, fetches: list[str]) -> None:
    """Stub edge_tts so every network fetch is recorded and none leaves the box."""
    import sys
    import types

    class _Counting:
        def __init__(self, text, voice, rate=None) -> None:
            fetches.append(text)

        async def stream(self):
            yield {"type": "audio", "data": _audio_bytes(np.array([0.5], dtype=np.float32), 24_000)}

    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=_Counting))


def test_a_clause_synthesized_by_an_earlier_run_is_not_refetched(monkeypatch, tmp_path) -> None:
    """The reason the disk cache exists. Measured over the 208 recorded calls in
    call_events.db, 67.9% of spoken clauses are text an EARLIER CALL already
    synthesized - and the process-local dict threw every one of them away at
    restart, so each one cost another network round trip (3-9s this session,
    30-50% of them timing out). A second process must find them on disk."""
    fetches: list[str] = []
    _counting_edge(monkeypatch, fetches)

    first = EdgeTts(TtsSettings(cache_dir=str(tmp_path), engine="edge", language="ta"))
    first._voice = "ta-IN-PallaviNeural"
    first.synthesize("வணக்கம்", "ta")

    # A whole new adapter, as a restarted server has: empty dict, same disk.
    second = EdgeTts(TtsSettings(cache_dir=str(tmp_path), engine="edge", language="ta"))
    second._voice = "ta-IN-PallaviNeural"
    result = second.synthesize("வணக்கம்", "ta")

    assert fetches == ["வணக்கம்"], f"the endpoint was hit {len(fetches)} times across two runs"
    assert result.samples.size > 0, "the cached body did not decode to audio"


def test_retuning_the_speaking_rate_does_not_serve_the_old_audio(monkeypatch, tmp_path) -> None:
    """The cache key carries voice and rate. Without that, changing TTS_RATE
    would leave every previously-cached clause speaking at the old speed for as
    long as the cache directory survived - a silent, very confusing bug."""
    fetches: list[str] = []
    _counting_edge(monkeypatch, fetches)

    for rate in ("+10%", "+25%"):
        tts = EdgeTts(TtsSettings(cache_dir=str(tmp_path), engine="edge", language="ta", rate=rate))
        tts._voice = "ta-IN-PallaviNeural"
        tts.synthesize("வணக்கம்", "ta")

    assert len(fetches) == 2, "a rate change reused audio synthesized at the old rate"


def test_a_clause_truncated_by_the_timeout_never_reaches_the_disk(monkeypatch, tmp_path) -> None:
    """The in-memory half of this is guarded above. On disk it matters more: a
    clipped body written to the cache outlives the process that wrote it and is
    replayed clipped by every later run, forever."""
    import sys
    import types

    body = _audio_bytes(np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32), 24_000)

    class _StallsMidBody:
        def __init__(self, text, voice, rate=None) -> None:
            pass

        async def stream(self):
            yield {"type": "audio", "data": body}
            await asyncio.sleep(5)

    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=_StallsMidBody))
    tts = EdgeTts(TtsSettings(cache_dir=str(tmp_path), timeout_seconds=0.2))
    tts._voice = "ta-IN-PallaviNeural"

    tts.synthesize("வணக்கம்", "ta")

    assert list(tmp_path.glob("*.mp3")) == [], "a truncated clause was written to the disk cache"


def test_an_unwritable_cache_directory_costs_a_round_trip_and_nothing_else(monkeypatch, tmp_path) -> None:
    """A full disk or a read-only mount must not take a call down. The cache is
    an optimisation in front of a working network path; every failure in it is
    allowed to cost latency and nothing more."""
    fetches: list[str] = []
    _counting_edge(monkeypatch, fetches)

    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")  # mkdir and write both fail on this

    tts = EdgeTts(TtsSettings(cache_dir=str(blocked), engine="edge", language="ta"))
    tts._voice = "ta-IN-PallaviNeural"

    result = tts.synthesize("வணக்கம்", "ta")

    assert result.samples.size > 0, "an unwritable cache directory broke synthesis"
    assert fetches == ["வணக்கம்"]


def test_an_english_word_glued_to_a_tamil_suffix_still_reaches_the_voice() -> None:
    """Reported live: "the TTS is not reading english words".

    Written Tamil-English code-mix glues an English word to its Tamil case
    suffix with a hyphen - "Cardiology-ல", "department-க்கு" - and the prompt
    teaches that style, so it is in almost every reply. The Tamil neural voice
    silently DROPS the English half of such a token. Measured by voiced-audio
    duration (total duration is useless - edge pads short clips to ~1.78s):

        department          0.68s   spoken
        department-க்கு      0.30s   the English word is GONE
        department க்கு      0.80s   spoken

    Splitting that one hyphen restored 51-60% more voiced audio on real agent
    lines. It is speech-only: the transcript keeps the hyphen.
    """
    from .tts import speakable

    assert speakable("Cardiology-ல appointment") == "Cardiology ல appointment"
    assert speakable("எந்த department-க்கு வேணும்?") == "எந்த department க்கு வேணும்?"
    assert speakable("Desk-ல இருந்து") == "Desk ல இருந்து"

    # Latin-to-Latin hyphens are real words, not case suffixes - leave them be.
    assert speakable("pre-auth follow-up co-pay") == "pre-auth follow-up co-pay"
    # ...and an identifier must survive intact or the caller hears a wrong ID.
    assert speakable("IP-2025-91043") == "IP-2025-91043"


def test_the_voice_is_given_the_speakable_text_not_the_written_text(monkeypatch) -> None:
    """The transform is worthless if synthesize() forgets to apply it."""
    import sys
    import types

    seen: list[str] = []

    class _Recording:
        def __init__(self, text, voice, rate=None) -> None:
            seen.append(text)

        async def stream(self):
            yield {"type": "audio", "data": _audio_bytes(np.array([0.5], dtype=np.float32), 24_000)}

    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=_Recording))
    tts = EdgeTts(TtsSettings(cache_dir="", engine="edge", language="ta"))
    tts._voice = "ta-IN-PallaviNeural"

    tts.synthesize("Cardiology-ல appointment book பண்ணணும்", "ta")

    assert seen == ["Cardiology ல appointment book பண்ணணும்"], (
        f"the engine was handed {seen!r} - the English word will be dropped"
    )


def test_the_configured_speaking_rate_reaches_the_engine(monkeypatch) -> None:
    """The default voice is slow enough to read as a recording. Measured on a
    real reply: +0% is 6.12s of audio, +15% is 5.33s, +25% is 4.92s. +15% is
    brisk without rushing, which matters when callers are elderly or anxious.

    It is also a latency win - every turn finishes sooner - so a rate that
    silently fails to reach edge would be an invisible regression."""
    import sys
    import types

    seen: dict[str, object] = {}

    class _Recording:
        def __init__(self, text, voice, rate=None) -> None:
            seen["rate"] = rate

        async def stream(self):
            yield {"type": "audio", "data": _audio_bytes(np.array([0.5], dtype=np.float32), 24_000)}

    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=_Recording))
    tts = EdgeTts(TtsSettings(cache_dir="", engine="edge", language="ta", rate="+20%"))
    tts._voice = "ta-IN-PallaviNeural"

    tts.synthesize("வணக்கம்", "ta")

    assert seen["rate"] == "+20%", f"edge was given rate={seen['rate']!r}"


def test_a_malformed_rate_is_rejected_at_startup_not_mid_call() -> None:
    """edge rejects anything that is not exactly +N%/-N%, and it does so on the
    first synthesis - i.e. mid-call, as silence."""
    import pytest

    for bad in ("fast", "15%", "+15", "++15%"):
        with pytest.raises(ValueError):
            TtsSettings(rate=bad)


# --- edge's silence padding, which the caller hears as a gap after every '.' ---
#
# Measured on real agent clauses at +15%: every clip opens with ~0.17s of
# silence and a SHORT clip is padded out to exactly 1.78s however brief it is
# ("சரி சார்." = 0.72s of speech, 0.16s in front, 0.90s behind). Clauses are
# synthesized separately and played back to back, so that padding accumulates
# at precisely the clause boundaries. Trimming those six clauses took the turn
# from 15.44s of audio to 12.30s.


def _padded_clip(sample_rate: int, lead: float, speech: float, trail: float) -> np.ndarray:
    """Silence, then a tone, then silence - the shape edge actually returns."""
    tone = 0.6 * np.sin(
        2 * np.pi * 220 * np.arange(int(speech * sample_rate)) / sample_rate
    ).astype(np.float32)
    return np.concatenate(
        [
            np.zeros(int(lead * sample_rate), dtype=np.float32),
            tone,
            np.zeros(int(trail * sample_rate), dtype=np.float32),
        ]
    )


def test_the_gap_after_a_full_stop_is_trimmed_to_the_configured_pause(monkeypatch) -> None:
    rate = 24_000
    settings = TtsSettings(cache_dir="", engine="edge", language="ta")
    tts = EdgeTts(settings)
    tts.load()
    _stub_mp3(monkeypatch, tts, _padded_clip(rate, lead=0.16, speech=0.72, trail=0.90), rate)

    result = tts.synthesize("சரி சார்.", "ta")

    duration = len(result.samples) / result.sample_rate
    expected = settings.clause_lead_seconds + 0.72 + settings.clause_pause_seconds
    # One 20ms detection frame of slack at each edge.
    assert abs(duration - expected) < 0.05, f"{duration:.3f}s, expected ~{expected:.3f}s"
    # The whole point: what used to be ~1.06s of dead air between clauses is now
    # one deliberate pause.
    assert duration < 1.78


def test_trimming_never_eats_the_speech_itself(monkeypatch) -> None:
    """A trim that clipped the attack of a word would be worse than the gap."""
    rate = 24_000
    settings = TtsSettings(cache_dir="", engine="edge", language="ta")
    tts = EdgeTts(settings)
    tts.load()
    _stub_mp3(monkeypatch, tts, _padded_clip(rate, lead=0.16, speech=0.72, trail=0.90), rate)

    result = tts.synthesize("சரி சார்.", "ta")

    loud = np.flatnonzero(np.abs(result.samples.astype(np.float32) / 32767.0) > 0.01)
    voiced = (loud[-1] - loud[0] + 1) / result.sample_rate
    assert voiced >= 0.70, f"only {voiced:.3f}s of the 0.72s of speech survived"
    # And the lead-in is kept, not cut flush - cutting flush clips a plosive.
    assert loud[0] > 0


def test_an_all_silent_clip_is_left_alone(monkeypatch) -> None:
    """A silent clause is a TTS failure, not something to turn into an empty array."""
    rate = 24_000
    tts = EdgeTts(TtsSettings(cache_dir="", engine="edge", language="ta"))
    tts.load()
    silence = np.zeros(int(0.5 * rate), dtype=np.float32)
    _stub_mp3(monkeypatch, tts, silence, rate)

    result = tts.synthesize("சரி சார்.", "ta")

    assert len(result.samples) == int(0.5 * rate)
