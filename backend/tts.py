"""TTS adapters. Two engines behind one interface, chosen by TTS_ENGINE.

`edge` (EdgeTts) is the working default: Microsoft's neural Tamil voices, no
API key, no model download, and it runs on CPU - which is what makes a voice
demo possible on this hardware at all. ta-IN-PallaviNeural is female.

PRIVACY - read before pointing this at real callers. edge sends the text to be
spoken to a Microsoft endpoint. That is fine for local testing with the
fictional Aruvi data in golden/, and NOT fine for real patient speech: this is
a hospital agent and the text is PHI-adjacent (see BACKEND_COMPLETION.md Sec4,
which already flags exactly this gap). For deployment use a self-hosted engine
- AI4Bharat Indic Parler-TTS is the recommended first try (SETUP.md) - behind
the same interface, and the rest of the pipeline does not change.

`svara` (SvaraTts) is the original placeholder, kept because it is what
BACKEND_COMPLETION.md Sec3.2 specifies; its load() still raises until a real
svara-TTS reference exists.

Both keep asr.py's load()/ready shape - loaded once at startup (see main.py's
lifespan) so per-call synthesis never pays model-load latency - and both return
mono int16 PCM at the engine's native rate. Resampling stays out of these
adapters so one code path serves a 16 kHz browser socket today and an 8 kHz SIP
leg later without knowing which it is talking to.

Original module docstring follows.

svara-TTS streaming adapter (voice-cloned, sentence-chunked synthesis).

Mirrors asr.py's load()/ready shape: the model and cloned-voice reference are
meant to be loaded once at startup (see main.py's lifespan) so per-call
synthesis never pays model-load latency. Output stays at the model's native
sample rate - BACKEND_COMPLETION.md Sec3.2 keeps resampling out of this
adapter so the same code path can serve a 16 kHz browser socket today and an
8 kHz SIP leg later without knowing which transport it's talking to.

load() is a placeholder: no real svara-TTS package/API reference exists yet.
synthesize()'s call into self._model assumes a plausible shape (text in,
float32 waveform in [-1, 1] out, at whatever rate self._model reports) so the
rest of this class - and main.py's wiring - is real and testable now; that
assumed shape is the only thing likely to need adjusting once the real
package lands.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import io
import logging
import os
from pathlib import Path
import re

import numpy as np

from .settings import TtsSettings

logger = logging.getLogger("aica.tts")


@dataclass(frozen=True)
class SynthesisResult:
    samples: np.ndarray  # mono int16 PCM at `sample_rate`
    sample_rate: int


# One retry, briefly delayed. Enough to ride out a connect timeout without
# adding a perceptible stall to a turn that was going to fail anyway.
EDGE_ATTEMPTS = 2
EDGE_RETRY_DELAY_SECONDS = 0.4

# Repeated clause text is the common case here, not a rare one worth
# optimising for: EVERY call opens with the same four greeting clauses and
# every silent tool turn says the same holding line. Caching the fetched MP3
# makes those instant (measured 0.89s -> 0s for the greeting) and, more
# importantly, makes them survive the endpoint being unreachable - which is
# the failure that otherwise leaves a caller with no voice at all.
#
# ponytail: in-memory and unbounded-LRU-free - a plain dict with a cap, since
# a call's vocabulary of repeated lines is tiny. Backed by a disk cache below,
# so this only has to be the hot layer.
EDGE_CACHE_MAX_ENTRIES = 64

# ...and the disk layer, which is the one that matters. Measured over the 208
# real recorded calls in call_events.db, 2505 spoken clauses:
#
#     repeat WITHIN a call     12.0%   the dict above already caught these
#     repeat ACROSS calls      67.9%   every one of these was a fresh network call
#     genuinely novel          20.1%
#
# So four clauses in five are text some earlier call already paid for, and the
# process-local dict threw all of it away at every restart. That is the whole
# TTS latency story on this deployment: the engine is a network round-trip to
# Microsoft, measured this session at a 3-9s median per novel clause with
# 30-50% of them timing out outright, and no amount of concurrency or timeout
# tuning moved it (both were measured and both were left alone). Not making the
# call at all is the only thing that does.
#
# Keyed on voice+rate+text so retuning TTS_RATE cannot serve yesterday's audio,
# and storing the MP3 body rather than decoded PCM so the clip stays reusable
# if the inter-clause pause is retuned.
#
# ponytail: no eviction. A hospital desk's spoken vocabulary is small and
# bounded by the prompt - the 208 recorded calls hold ~800 distinct clauses at
# ~20KB each, so ~16MB. Add an age sweep if a long-lived deployment proves that
# wrong.
_CACHE_VERSION = "v1"

# Written Tamil-English code-mix glues an English word to its Tamil case
# suffix with a hyphen - "Cardiology-ல", "department-க்கு", "bill-ல" - and the
# prompt teaches exactly that style, so it is in almost every reply.
#
# The Tamil neural voice SILENTLY DROPS the English half of such a token.
# Measured, voiced-audio duration (total duration is useless here, edge pads
# every short clip to ~1.78s):
#
#     department          0.68s   spoken
#     department-க்கு      0.30s   the English word is GONE, only "க்கு" is said
#     department க்கு      0.80s   spoken
#     Cardiology          0.82s   spoken
#     Cardiology-ல         0.28s   GONE
#     Cardiology ல         1.10s   spoken
#
# So the hyphen, not the script mixing, is what breaks it. Replacing just that
# one hyphen with a space restores the word. This is a SPEECH-ONLY transform:
# the transcript, the call log and the model's own history keep the hyphen,
# which is how the language is actually written.
#
# Latin-to-Latin hyphens ("pre-auth", "co-pay", "follow-up") are untouched -
# the pattern requires a Tamil character on the right-hand side.
_LATIN_TAMIL_HYPHEN_RE = re.compile(r"([A-Za-z0-9])-(?=[\u0b80-\u0bff])")


def speakable(text: str) -> str:
    """Rewrite one clause into what the voice can actually pronounce."""
    return _LATIN_TAMIL_HYPHEN_RE.sub(r"\1 ", text)


class Mp3DiskCache:
    """Synthesized MP3 bodies, kept across restarts. Never raises.

    Every method swallows OS errors deliberately: this is a latency
    optimisation sitting in front of a working network path, so a full disk, a
    read-only mount or a permissions problem must cost a round trip and
    nothing else. A cache that can take down a call is worse than no cache.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)

    def _path(self, voice: str, rate: str, text: str) -> Path:
        digest = hashlib.sha256(
            "\x00".join((_CACHE_VERSION, voice, rate, text)).encode("utf-8")
        ).hexdigest()
        return self.directory / f"{digest}.mp3"

    def get(self, voice: str, rate: str, text: str) -> bytes | None:
        try:
            body = self._path(voice, rate, text).read_bytes()
        except OSError:
            return None
        # An empty file is a failed write, not a silent clause: treat it as a
        # miss so it gets re-fetched rather than serving silence forever.
        return body or None

    def put(self, voice: str, rate: str, text: str, body: bytes) -> None:
        if not body:
            return
        path = self._path(voice, rate, text)
        # Written to a unique temp name and renamed, because os.replace is
        # atomic: a crash or a second worker mid-write can otherwise leave a
        # TRUNCATED mp3 that every later call happily serves as a clipped
        # clause, forever. The pid suffix keeps two processes from colliding
        # on the same temp file.
        temp = path.with_suffix(f".{os.getpid()}.part")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            temp.write_bytes(body)
            os.replace(temp, path)
        except OSError as error:
            logger.warning("TTS disk cache write failed for %r: %s", text[:40], error)
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass


def trim_padding(
    samples: np.ndarray,
    sample_rate: int,
    lead_seconds: float,
    pause_seconds: float,
    threshold: float,
) -> np.ndarray:
    """Cut edge's silent padding down to one deliberate inter-clause pause.

    Edge returns every clip padded, and a SHORT clip is padded hardest -
    measured, a clause is padded out to exactly 1.78s however brief it is, so
    "சரி சார்." arrives as 0.72s of speech behind 0.16s of silence
    and in front of 0.90s more. A turn's clauses are synthesized separately and
    played back to back, so all of that padding lands at the clause boundaries
    and the caller hears a long gap after every full stop.

    Detects speech on 20ms frame peaks rather than a per-sample threshold: a
    single sample crossing it mid-silence would otherwise defeat the whole
    trim, and Tamil word-final vowels trail off quietly enough that a
    per-sample test on a soft frame reads as speech.

    Returns the input untouched if nothing crosses the threshold - a clause
    that is genuinely all silence is a TTS failure, and this is not the place
    to turn it into an empty array the sender would treat differently.
    """
    if samples.size == 0:
        return samples

    frame = max(1, int(sample_rate * 0.02))
    usable = samples.size // frame * frame
    if usable == 0:
        return samples

    peaks = np.abs(samples[:usable].astype(np.float32) / 32767.0).reshape(-1, frame).max(axis=1)
    voiced = np.flatnonzero(peaks > threshold)
    if voiced.size == 0:
        return samples

    start = max(0, int((voiced[0] * frame) - lead_seconds * sample_rate))
    end = min(samples.size, int((voiced[-1] + 1) * frame))
    pause = np.zeros(int(pause_seconds * sample_rate), dtype=samples.dtype)
    return np.concatenate([samples[start:end], pause])


# Microsoft's neural voices for the languages settings.py already accepts.
# Female by default: the golden transcripts are written around a woman at the
# desk (Gayathri, Deepa, Kavitha...), and conversation.OPENING_LINE names her.
_EDGE_VOICES: dict[str, str] = {
    "ta": "ta-IN-PallaviNeural",
    "hi": "hi-IN-SwaraNeural",
    "te": "te-IN-ShrutiNeural",
    "ml": "ml-IN-SobhanaNeural",
    "kn": "kn-IN-SapnaNeural",
    "bn": "bn-IN-TanishaaNeural",
    "mr": "mr-IN-AarohiNeural",
    "gu": "gu-IN-DhwaniNeural",
    "pa": "pa-IN-OjasNeural",
}


def _run_blocking(coro):
    """Run `coro` to completion from synchronous code, loop or no loop.

    synthesize() is a sync method with an async engine underneath. main.py
    reaches it through asyncio.to_thread, where the thread has no loop and
    asyncio.run() is fine - but any caller that is itself async (the transcript
    logger, a notebook, a future streaming path) would hit "asyncio.run()
    cannot be called from a running event loop". Handing the coroutine to a
    private loop on its own thread works in both cases.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class EdgeTts:
    """Microsoft neural TTS. No API key, no model download, runs on CPU.

    See this module's docstring for the privacy constraint before using it with
    anything but the fictional test data.
    """

    def __init__(self, settings: TtsSettings) -> None:
        self.settings = settings
        self._voice: str | None = None
        # text -> complete MP3 body. Only ever written from the success path
        # in _stream_mp3(), never from the partial-on-timeout one, so a
        # clipped clause can't be served for the rest of the process's life.
        self._mp3_cache: dict[str, bytes] = {}
        # Survives restarts; see _CACHE_VERSION for why it is the whole point.
        self._disk_cache = Mp3DiskCache(settings.cache_dir) if settings.cache_dir else None
        # edge streams MP3; libsndfile 1.2+ decodes it, and every clip so far
        # has come back at 24 kHz. Read from the decoder rather than assumed,
        # since main.py tells the client the rate before the first clause.
        self._sample_rate: int = 24_000

    @property
    def ready(self) -> bool:
        return self._voice is not None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def load(self) -> None:
        import edge_tts  # noqa: F401  - fail loudly at startup, not mid-call
        import soundfile

        if "MP3" not in soundfile.available_formats():
            raise RuntimeError(
                "libsndfile has no MP3 decoder (needs >= 1.1); "
                f"found {soundfile.__libsndfile_version__}. pip install -U soundfile"
            )

        voice = self.settings.voice or _EDGE_VOICES.get(self.settings.language)
        if voice is None:
            raise RuntimeError(f"no edge voice mapped for language {self.settings.language!r}")

        self._voice = voice
        logger.info("edge TTS ready: voice=%s rate=%s", voice, self.settings.rate)

    def synthesize(self, text: str, language: str) -> SynthesisResult:
        if self._voice is None:
            raise RuntimeError("TTS model is not loaded")

        text = text.strip()
        if not text:
            return SynthesisResult(np.array([], dtype=np.int16), self._sample_rate)

        mp3 = _run_blocking(self._stream_mp3(speakable(text)))
        if not mp3:
            logger.warning("edge TTS returned no audio for %r", text[:60])
            return SynthesisResult(np.array([], dtype=np.int16), self._sample_rate)

        import soundfile

        waveform, sample_rate = soundfile.read(io.BytesIO(mp3), dtype="float32", always_2d=True)
        self._sample_rate = int(sample_rate)
        mono = waveform.mean(axis=1)
        samples = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
        # After decode, not before: the MP3 cache stores the body edge sent, so
        # the same clip stays reusable if the pause is ever retuned.
        samples = trim_padding(
            samples,
            self._sample_rate,
            self.settings.clause_lead_seconds,
            self.settings.clause_pause_seconds,
            self.settings.silence_threshold,
        )
        return SynthesisResult(samples=samples, sample_rate=self._sample_rate)

    def _remember(self, text: str, body: bytes) -> None:
        """Record one COMPLETE clause body in both cache layers.

        Only ever called from the success path in _stream_mp3(), never from the
        partial-on-timeout one - a clipped clause must not be served for the
        rest of the process's life, still less written to disk where it would
        outlive it.
        """
        if not body:
            return
        if len(self._mp3_cache) < EDGE_CACHE_MAX_ENTRIES:
            self._mp3_cache[text] = body
        if self._disk_cache is not None and self._voice:
            self._disk_cache.put(self._voice, self.settings.rate, text, body)

    async def _stream_mp3(self, text: str) -> bytes:
        """Fetch one clause's audio, retrying a transient network failure.

        This engine is a network call to a Microsoft endpoint, so a single
        dropped connection is an ordinary event, not a fault - and losing it
        costs the caller the voice for that clause while the text still goes
        out, which reads as the agent going silent mid-sentence. One quick
        retry converts the common case (a timeout on connect) into a small
        delay. A second failure is reported to the caller of synthesize()
        rather than retried further: on a voice channel, late audio is worth
        less than a turn that moves on.
        """
        import edge_tts

        cached = self._mp3_cache.get(text)
        if cached is not None:
            return cached

        if self._disk_cache is not None:
            cached = self._disk_cache.get(self._voice, self.settings.rate, text)
            if cached is not None:
                # Promoted into the hot dict so a line repeated inside one call
                # does not go back to the filesystem for every clause.
                self._remember(text, cached)
                return cached

        timeout = self.settings.timeout_seconds
        last_error: Exception | None = None
        for attempt in range(EDGE_ATTEMPTS):
            # Outside the try: a timeout must be able to keep whatever already
            # arrived. MP3 is a stream of independent frames, so a truncated
            # body still decodes to the part of the clause that got through -
            # a clipped word is worth more to a caller than silence, and the
            # old code threw all of it away.
            chunks = bytearray()
            try:
                async with asyncio.timeout(timeout):
                    stream = edge_tts.Communicate(text, self._voice, rate=self.settings.rate).stream()
                    async for chunk in stream:
                        if chunk["type"] == "audio":
                            chunks += chunk["data"]
                    body = bytes(chunks)
                    self._remember(text, body)
                    return body
            except TimeoutError as error:
                # Deliberately NOT retried. A timeout means the endpoint is
                # stalling rather than refusing, and the second attempt stalls
                # the same way: measured 10s + 10s = 21s on one clause, during
                # which the whole turn's remaining audio sat behind it.
                if chunks:
                    logger.warning(
                        "edge TTS timed out after %.1fs for %r - keeping the %d bytes that arrived",
                        timeout, text[:40], len(chunks),
                    )
                    return bytes(chunks)
                logger.warning("edge TTS timed out after %.1fs for %r", timeout, text[:40])
                raise RuntimeError(f"edge TTS timed out after {timeout:.1f}s") from error
            except Exception as error:  # aiohttp raises a family of these
                last_error = error
                logger.warning(
                    "edge TTS attempt %d/%d failed for %r: %s",
                    attempt + 1,
                    EDGE_ATTEMPTS,
                    text[:40],
                    error,
                )
                if attempt + 1 < EDGE_ATTEMPTS:
                    await asyncio.sleep(EDGE_RETRY_DELAY_SECONDS)

        raise RuntimeError(f"edge TTS failed after {EDGE_ATTEMPTS} attempts: {last_error}")


class SvaraTts:
    """Loads the svara-TTS model + cloned-voice reference once and synthesizes text."""

    def __init__(self, settings: TtsSettings) -> None:
        self.settings = settings
        self._model = None
        self._sample_rate: int | None = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def sample_rate(self) -> int:
        if self._sample_rate is None:
            raise RuntimeError("TTS model is not loaded")
        return self._sample_rate

    def load(self) -> None:
        """Load model weights and the cloned-voice reference during startup.

        PLACEHOLDER: replace this method's body with the real svara-TTS model
        + voice-reference load once that reference is available. Everything
        else in this class - ready gating, synthesize()'s output shape,
        main.py's wiring - is written against the interface below and should
        not need to change when this does.
        """
        raise NotImplementedError(
            "SvaraTts.load() is a placeholder - wire in the real svara-TTS model "
            "and cloned-voice reference here (see backend/tts.py)."
        )

    def synthesize(self, text: str, language: str) -> SynthesisResult:
        """Return mono int16 PCM for one clause, at the model's native rate."""
        if self._model is None:
            raise RuntimeError("TTS model is not loaded")

        text = text.strip()
        if not text:
            return SynthesisResult(samples=np.array([], dtype=np.int16), sample_rate=self.sample_rate)

        waveform = np.asarray(
            self._model.synthesize(
                text, language=language, voice_reference_path=self.settings.voice_reference_path
            ),
            dtype=np.float32,
        )
        samples = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
        return SynthesisResult(samples=samples, sample_rate=self.sample_rate)


def create_tts(settings: TtsSettings):
    """Return the adapter named by TTS_ENGINE.

    main.py holds this as `SvaraTts` for typing purposes only - both adapters
    expose the same load()/ready/sample_rate/synthesize surface, which is the
    whole point of selecting between them here rather than at the call site.
    """
    if settings.engine == "edge":
        return EdgeTts(settings)
    if settings.engine == "svara":
        return SvaraTts(settings)
    raise ValueError(f"unknown TTS_ENGINE: {settings.engine!r} (expected 'edge' or 'svara')")
