"""Runtime settings for the low-latency audio pipeline.

Defaults are the hyperparameters from README.md, which is the configuration
the live mic pipeline was actually tuned on - do not "improve" them casually.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from pathlib import Path

from dotenv import load_dotenv

# .env (gitignored) holds HF_TOKEN and every override in this file - see
# .env.example, which documents all of them.
#
# The load that MATTERS is in backend/__init__.py, not here. This module is not
# the first thing imported: clause_chunker, conversation and main all have
# module-level os.getenv() calls that run at IMPORT time, and main.py imports
# them before it imports this file - so a .env loaded here would arrive after
# they had already read their defaults. Loading from the package root fixes
# that for every module at once.
#
# Kept anyway, and it is a no-op in the normal path: load_dotenv does not
# override a name already in the environment, so this only ever matters if
# settings.py is somehow reached without backend/__init__.py having run.
load_dotenv()


# Languages IndicConformer accepts as `language_id`.
SUPPORTED_LANGUAGES = frozenset({"ta", "hi", "te", "ml", "kn", "bn", "mr", "gu", "pa"})


@dataclass(frozen=True)
class AudioSettings:
    """Tuned for conversational turn-taking, not long-form batch transcription."""

    # Must stay 16 kHz: both TEN VAD and IndicConformer are trained on it.
    sample_rate: int = 16_000
    vad_hop_size: int = 256  # 16 ms at 16 kHz; TEN VAD is tuned around this.

    # Lower = catches softer/quieter onsets but more false triggers on noise.
    vad_threshold: float = float(os.getenv("VAD_THRESHOLD", "0.35"))

    # TEN VAD flags speech per 16 ms hop, and ONE positive hop is not a turn.
    # Measured over the 87 real captured turns in call_events.db, 46 of them
    # transcribed to 3 characters or fewer and 34 to the EMPTY STRING - i.e.
    # 39% of everything the VAD opened contained no speech at all. Requiring
    # 4 consecutive positive hops (64 ms) before a turn opens removes the
    # shortest of those without costing a real caller anything: the candidate
    # frames are kept and prepended, so confirming an onset never clips the
    # first syllable.
    #
    vad_start_frames: int = int(os.getenv("VAD_START_FRAMES", "4"))

    # LOUDNESS AT ONSET. Only speech this loud may OPEN a turn, so a
    # television, a fan or a conversation across the room no longer starts one.
    #
    # Read this before touching it. An energy gate was added here once before
    # and reverted, because it gated EVERY frame: a quiet trailing syllable
    # scored as "silence", the endpoint countdown ran on through the middle of
    # a word, and turns came back as one-character transcripts ("ந", "ப", "க").
    # The lesson was not that loudness is unusable, it is that loudness must
    # never be allowed to END a turn - only to refuse to start one. So these
    # two knobs are read in exactly one place, the not-yet-in-speech branch of
    # TenVadSegmenter.process(). Once a turn is open, endpointing is decided by
    # the VAD flag alone, and no quiet syllable can cut it short.
    #
    # Measured, per 16 ms frame of real Tamil speech at full digital level:
    #
    #     voiced p10   931      voiced p50  2624     peak  9804
    #
    # so 200 sits about 4.6x below the quietest speech frame - low enough to
    # be a backstop rather than a filter, because microphone gain varies wildly
    # between machines and an absolute number cannot be right for all of them.
    # The SNR term is the part that actually adapts: the room's noise floor is
    # learned continuously while nobody is speaking, and onset has to beat a
    # multiple of it. Raise VAD_ONSET_SNR first if a noisy room still opens
    # turns; raise VAD_ONSET_MIN_RMS only if the microphone is unusually hot.
    vad_onset_min_rms: float = float(os.getenv("VAD_ONSET_MIN_RMS", "200"))
    vad_onset_snr: float = float(os.getenv("VAD_ONSET_SNR", "3.0"))

    # THE WATCHDOG. A turn ends when nothing LOUD has arrived for this long,
    # whatever the VAD is flagging. 44 x 16 ms = 704 ms.
    #
    # Without it a turn can be held open forever, and this is not theoretical:
    # the endpoint countdown is restarted by any sustained run of flagged
    # frames, and once the agent has spoken once, residual echo and the
    # browser's automatic gain control produce exactly such runs out of an
    # empty room. Observed live - the first turn of a call endpointed normally
    # and every turn after it listened until the 30 s hard cap.
    #
    # Deliberately TWICE endpoint_silence_frames, not equal to it. Setting the
    # two equal would make a quiet flagged frame identical to silence, which is
    # the reverted behaviour that cut turns off mid-word. Double gives a
    # trailing syllable room to be quiet without giving noise room to hold the
    # microphone open. Someone actually speaking refreshes this constantly.
    vad_quiet_endpoint_frames: int = int(os.getenv("VAD_QUIET_ENDPOINT_FRAMES", "44"))

    # How fast the learned room-noise floor follows the room. Updated only
    # while no turn is open AND the VAD says the frame is not speech, so a
    # caller talking can never raise the bar against themselves.
    vad_noise_ema: float = float(os.getenv("VAD_NOISE_EMA", "0.05"))

    # Once the endpoint countdown has begun, only a sustained run of speech
    # restarts it. Resetting on a SINGLE positive hop is what let background
    # noise hold the microphone open indefinitely - one blip inside every
    # 352 ms window and the turn never closes, which is why a caller ends up
    # toggling their mic by hand after speaking. Real speech produces long
    # runs and clears this instantly; an isolated blip now costs one frame.
    vad_resume_frames: int = int(os.getenv("VAD_RESUME_FRAMES", "3"))

    # 8 x 16 ms = 128 ms kept *before* VAD fires, so VAD onset lag doesn't
    # clip the first syllable.
    pre_roll_frames: int = int(os.getenv("VAD_PRE_ROLL_FRAMES", "8"))

    # 22 x 16 ms = 352 ms of trailing silence before the turn closes. This sits
    # directly in front of every reply the caller hears - it is spent before the
    # ASR has even started - so it is part of the latency budget, not free.
    #
    # Was 30 (480 ms). Lowered because 480 ms is past the point where a listener
    # starts to feel talked-at-by-a-machine: human turn-taking gaps cluster
    # around 200 ms. Not lowered further than 352 ms on purpose - below roughly
    # 300 ms a natural mid-sentence breath starts closing the turn, and the ASR
    # gets half a sentence, which costs far more than the 150 ms it saves.
    #
    # Raise it back with VAD_ENDPOINT_SILENCE_FRAMES if callers who pause to
    # think are being cut off; that failure looks like the agent answering a
    # question the caller had not finished asking.
    endpoint_silence_frames: int = int(os.getenv("VAD_ENDPOINT_SILENCE_FRAMES", "22"))

    # 15 x 16 ms = 240 ms of continuous speech before the caller is allowed to
    # cut the agent off. Barging in on the first flagged hop meant any cough,
    # keystroke or breath near the mic stopped the agent mid-sentence. A real
    # interjection is a spoken word - several hundred ms - so it still lands
    # promptly; noise blips are one or two hops and now pass under the gate.
    #
    # Lower it with VAD_BARGE_IN_FRAMES if interrupting feels sluggish; raise
    # it in a noisy room. This gates ONLY the interrupt: the utterance is still
    # captured and transcribed from its first frame either way.
    barge_in_speech_frames: int = int(os.getenv("VAD_BARGE_IN_FRAMES", "15"))

    # The bar to interrupt the agent WHILE ITS OWN VOICE IS STILL AUDIBLE.
    # 40 x 16 ms = 640 ms of unbroken speech.
    #
    # Echo cancellation is on in the browser and mostly works, but what leaks
    # through is the agent's own sentence - and to a VAD that is not "noise",
    # it is genuine sustained speech, just not the caller's. A consecutive-hop
    # gate therefore cannot separate the two on its own, and this is why a
    # caller who leaves the microphone open hears the agent cut itself off.
    #
    # Raising the bar can separate them, because the two behave differently:
    # room noise and residual echo arrive in bursts, while a caller who
    # actually wants the floor keeps talking. 640 ms is roughly two words - a
    # deliberate interruption, not a reaction.
    #
    # This is only the gate for CUTTING THE AGENT OFF. The caller's utterance
    # is still captured and transcribed from its first frame either way, so
    # nothing they say is lost while this is waiting.
    barge_in_frames_while_audible: int = int(os.getenv("VAD_BARGE_IN_FRAMES_WHILE_AUDIBLE", "40"))

    # ponytail: not in the reference CLI, which is a trusted local mic. A
    # browser socket is not: without a cap, a stuck speech flag buffers
    # forever. 30 s is far past any real turn, so it never truncates one.
    max_utterance_frames: int = int(os.getenv("ASR_MAX_UTTERANCE_FRAMES", "1875"))

    language: str = os.getenv("ASR_LANGUAGE", "ta")

    # rnnt is slower than ctc but more accurate on this hybrid model -
    # correctness over latency for transcript quality.
    decoding: str = os.getenv("ASR_DECODING", "rnnt")

    # BACKEND_COMPLETION.md Sec3.5: one global asr_lock serializes every
    # concurrent call's ASR work, which won't scale past a handful of calls.
    # This bounds concurrent ASR inference to whatever the GPU can actually
    # hold rather than either (a) one-at-a-time or (b) fully unbounded.
    asr_max_concurrency: int = int(os.getenv("ASR_MAX_CONCURRENCY", "2"))

    def __post_init__(self) -> None:
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported ASR_LANGUAGE: {self.language}")
        if self.decoding not in {"ctc", "rnnt"}:
            raise ValueError("ASR_DECODING must be either 'ctc' or 'rnnt'")
        if self.asr_max_concurrency <= 0:
            raise ValueError("ASR_MAX_CONCURRENCY must be positive")
        if self.vad_start_frames <= 0 or self.vad_resume_frames <= 0:
            raise ValueError("VAD_START_FRAMES and VAD_RESUME_FRAMES must be positive")
        if self.vad_quiet_endpoint_frames <= self.endpoint_silence_frames:
            raise ValueError(
                "VAD_QUIET_ENDPOINT_FRAMES must exceed VAD_ENDPOINT_SILENCE_FRAMES - "
                "equal makes a quiet syllable indistinguishable from silence"
            )
        if self.vad_onset_min_rms < 0 or self.vad_onset_snr < 1:
            raise ValueError("VAD_ONSET_MIN_RMS must be >= 0 and VAD_ONSET_SNR must be >= 1")
        if not 0 < self.vad_noise_ema <= 1:
            raise ValueError("VAD_NOISE_EMA must be in (0, 1]")
        if self.barge_in_frames_while_audible < self.barge_in_speech_frames:
            raise ValueError(
                "VAD_BARGE_IN_FRAMES_WHILE_AUDIBLE must be >= VAD_BARGE_IN_FRAMES - "
                "interrupting must never be EASIER while the agent's own voice is in the room"
            )


@dataclass(frozen=True)
class LlmSettings:
    """Config for the OpenAI-compatible LLM client (vLLM/TGI) - see BACKEND_COMPLETION.md Sec3.1.

    An OpenAI-compatible client is used rather than a hand-rolled transport so
    the model is swappable (Llama 3.1 8B -> Qwen2.5-7B -> a future model) via
    one base-URL/model-name env var, without touching llm.py or conversation.py.
    """

    # 127.0.0.1, never "localhost". Measured on this box: the same /api/chat
    # request takes 2.10s via localhost and 0.05s via 127.0.0.1 - name
    # resolution tries an address the server is not listening on and waits out
    # a timeout first. That 2s is paid on EVERY LLM call, and a turn with a
    # tool call makes several, so it dominated time-to-first-word.
    base_url: str = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8001/v1")
    api_key: str = os.getenv("LLM_API_KEY", "not-needed")
    model: str = os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

    # Low: this is a task-following hospital agent, not a creative one.
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    # Turns are capped at ~40 words by the master prompt's turn discipline
    # (section 3), so completions don't need much headroom.
    #
    # 300 -> 160. The longest LEGITIMATE turn measured on a real call is the
    # closing handoff at ~115 tokens, so 160 keeps 40% headroom over anything
    # the turn discipline permits. What 300 bought was runaway: a degenerate
    # generation ("... அடிப்படை அடிப்படை அடிப்படை ..." x26) ran the full 300
    # tokens and spoke 25 SECONDS of gibberish at a caller before the cap
    # stopped it. The cap is the only thing that ever stops that, so it should
    # sit just above real turns rather than far above them.
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "160"))

    # THE CONTEXT WINDOW, and it MUST match `PARAMETER num_ctx` in the
    # Modelfile. Ollama's window is set there and nothing at runtime can read
    # it back, so this is a restatement - which is exactly how the two drift.
    # conversation.py sizes its history trim against this number, so a value
    # LARGER than the Modelfile's silently re-creates the overflow that trim
    # exists to prevent: Ollama truncates from the FRONT, taking the system
    # prompt's language and clinical-safety rules with it, and says nothing.
    # test_the_worst_case_turn_still_fits_inside_num_ctx reads the Modelfile
    # and fails if they disagree.
    num_ctx: int = int(os.getenv("LLM_NUM_CTX", "6144"))

    # GPU OFFLOAD, as a layer count. Applied when the model is BUILT
    # (backend/scripts/setup_model.py), not per request - Ollama reads it from
    # the model's own parameters.
    #
    # Blank (the default) means "let llama.cpp decide", and that is not
    # timidity, it is a measured failure. `num_gpu 99` forces every layer onto
    # the card and measured 2.2x on a QUIET one (12.9 -> 31.3 tok/s, the
    # largest latency win on this project). But llama.cpp reserves ~1 GiB of
    # free VRAM, and when the projection misses that target it normally
    # RECOVERS by offloading fewer layers. An explicit num_gpu takes that
    # recovery away:
    #
    #     projected to use 2915 MiB vs 3344 MiB of free device memory
    #     cannot meet free memory target of 1024 MiB, reduce by 595 MiB
    #     failed to fit params: n_gpu_layers already set by user to 99, abort
    #
    # That is a 500 from Ollama on EVERY turn, not a slow turn - measured on
    # this 4 GB card with an ordinary desktop running. Auto-fit degrades to a
    # CPU/GPU split instead, which is slower and alive.
    #
    # Set LLM_NUM_GPU=99 to take the 2.2x back on a machine whose card is
    # genuinely free (close the Electron apps first, check `ollama ps` says
    # 100% GPU), and rebuild the model afterwards.
    num_gpu: str = os.getenv("LLM_NUM_GPU", "").strip()

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("LLM_MODEL must not be empty")
        if self.num_gpu and not self.num_gpu.isdigit():
            raise ValueError(f"LLM_NUM_GPU must be a layer count or blank, got {self.num_gpu!r}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 2")
        if self.max_tokens <= 0:
            raise ValueError("LLM_MAX_TOKENS must be positive")
        if self.num_ctx <= self.max_tokens:
            raise ValueError("LLM_NUM_CTX must leave room for LLM_MAX_TOKENS")


_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ConversationSettings:
    """Config for the Conversation Manager: master prompt location and tool-loop bounds."""

    # The full ~15k-token specification. It is NOT sent whole any more - only
    # its section-8 flow playbooks are parsed out of it, one per turn, by
    # backend/prompt_builder.py. It stays the single source of truth for those.
    prompt_path: Path = Path(
        os.getenv("CONVERSATION_PROMPT_PATH", str(_REPO_ROOT / "golden" / "main_prompt.txt"))
    )

    # The condensed core actually sent every turn (~1.5k tokens): language,
    # turn discipline, ledger, grounding, clinical safety, emergency override.
    runtime_core_path: Path = Path(
        os.getenv("CONVERSATION_RUNTIME_CORE_PATH", str(_REPO_ROOT / "golden" / "runtime_core.txt"))
    )

    # Two or three real exchanges per flow. Rules describe the register;
    # examples are what actually hold a small model in Tamil-English code-mix.
    exemplars_path: Path = Path(
        os.getenv("CONVERSATION_EXEMPLARS_PATH", str(_REPO_ROOT / "golden" / "flow_exemplars.json"))
    )

    # LLM_MAX_TOOL_ITERATIONS was here, capping LLM<->tool round-trips per
    # turn. It went with the tool layer (LLM_STACK.md Sec5) and was dead:
    # validated on every startup, read by nothing. Restoring tools per-flow
    # means restoring it - along with the 1778 prompt tokens and the eval
    # regression that measurement recorded.

    # v1 has no caller-metadata source (no CRM lookup, no SIP headers) wired
    # in yet, so this is the only {{agent_name}} substitution main.py can
    # supply at call_started - everything else (mrn, patient_name, ...) stays
    # blank until the LLM fills it in mid-call via lookupPatient et al.
    agent_name: str = os.getenv("CONVERSATION_AGENT_NAME", "Gayathri")

    def __post_init__(self) -> None:
        if not self.agent_name:
            raise ValueError("CONVERSATION_AGENT_NAME must not be empty")


@dataclass(frozen=True)
class TtsSettings:
    """Config for the svara-TTS adapter - see BACKEND_COMPLETION.md Sec3.2.

    model/voice_reference_path are placeholders: no real svara-TTS package
    reference exists yet, so backend/tts.py's load() raises NotImplementedError
    until one is wired in (see that file). Defined here now so the env-var
    surface doesn't change once it is.
    """

    # "edge" is the working default (Microsoft neural voices, no API key, CPU);
    # "svara" selects the placeholder adapter whose load() still raises. See
    # backend/tts.py's docstring for the privacy constraint on "edge".
    engine: str = os.getenv("TTS_ENGINE", "edge")

    # Blank means "pick the female voice mapped for `language`" - see
    # tts._EDGE_VOICES. Set this to override (e.g. ta-IN-ValluvarNeural, male).
    voice: str = os.getenv("TTS_VOICE", "")

    model: str = os.getenv("TTS_MODEL", "svara-tts")
    voice_reference_path: str = os.getenv("TTS_VOICE_REFERENCE_PATH", "")
    language: str = os.getenv("TTS_LANGUAGE", "ta")

    # BACKEND_COMPLETION.md Sec3.5: same global-lock scaling problem as ASR,
    # for the same reason (GPU-bound synthesis) - bound it instead of
    # serializing every call's TTS work behind one mutex.
    #
    # 4, not 2, because the DEFAULT engine is not GPU-bound at all: "edge" is a
    # network round-trip (see tts.py), so this bounds sockets rather than
    # compute, and main.py pipelines a turn's clauses through it to hide that
    # latency. Lower it to 2 if TTS_ENGINE is switched to a local GPU model.
    max_concurrency: int = int(os.getenv("TTS_MAX_CONCURRENCY", "4"))

    # A healthy "edge" clause synthesizes in ~0.9s (measured, sequential AND
    # 4-way concurrent). Past this the endpoint is stalling, and the sender is
    # holding every LATER clause's audio behind it (the queue is ordered), so
    # one stuck clause mutes the rest of the turn. Measured with the endpoint
    # unreachable: no timeout at all meant 21s per clause.
    #
    # Was 5s, and that was too tight. On a degraded link every clause of a turn
    # exceeded it and the caller got TOTAL SILENCE where the pre-timeout code
    # had given slow-but-present speech - observed live, 100% of clauses across
    # two turns. The bound has to be loose enough that "slow" still speaks.
    # 10s is ~11x the healthy median, and it is paid ONCE per turn rather than
    # once per clause: main.py synthesizes a turn's clauses concurrently, so
    # they stall in parallel (see the "TTS send waited 5.03s ... 0.00s ... 0.00s"
    # signature in the server log). tts.py also now keeps whatever bytes arrived
    # before the deadline instead of discarding them.
    timeout_seconds: float = float(os.getenv("TTS_TIMEOUT_SECONDS", "10"))

    # Speaking rate, as edge's own +N%/-N% string. A hospital desk agent talks
    # briskly; the default voice is slow enough to feel like a recording.
    # Measured on a real reply ("நன்றி முருகேசன் சார். உங்க registered mobile
    # number ஒரு தடவை சொல்லுங்களா?"):
    #
    #     +0%   6.12s      +20%  5.11s
    #     +10%  5.57s      +25%  4.92s
    #     +15%  5.33s      +30%  4.73s
    #
    # +10% is the slightly slower setting requested for clearer listening;
    # callers to a hospital are often elderly or anxious. Raise it with
    # TTS_RATE if the demo audience prefers faster; past about +25% the Tamil
    # starts to clip.
    #
    # This also shortens every turn, so it is a latency win as well as a
    # naturalness one: the caller stops waiting for the agent to finish sooner.
    # +10%, not +0%. This code default said +0% while HANDOFF.md Sec5 and
    # .env.example both documented +10%, so the shipped rate was whichever of
    # the three the reader happened to believe. +10% is the one that was
    # actually asked for and measured, and it shortens every turn by ~9%.
    rate: str = os.getenv("TTS_RATE", "+10%")

    # Edge PADS every clip it returns, and the padding is what the caller hears
    # as a long gap after every full stop. Measured on real agent clauses at
    # +15%, per clip:
    #
    #     total  lead  trail  voiced   clause
    #      1.78  0.18   0.76    0.84   கண்டிப்பா சார்.
    #      1.78  0.18   0.84    0.76   நன்றி சார்.
    #      1.78  0.16   0.90    0.72   சரி சார்.
    #      2.42  0.16   0.14    2.12   உங்க registered mobile number சொல்லுங்களா?
    #      5.90  0.18   0.14    5.58   எல்லாம் குறிச்சுக்கிட்டேன் — ...
    #
    # Every clip opens with ~0.17s of silence, and a SHORT one is padded out to
    # exactly 1.78s - so the shorter the clause, the longer the dead air behind
    # it. A turn's clauses are synthesized separately and played back to back,
    # so that padding accumulates at precisely the clause boundaries: 4.02s of
    # silence across those six clauses, ~0.67s per boundary.
    #
    # Trimming to a fixed, deliberate pause is what makes the speech sound
    # continuous. These are the two calibration knobs for it:
    #
    #   lead_trim  what is KEPT in front of the first voiced sample. Not zero:
    #              cutting flush to the threshold clips the attack of a plosive
    #              ("பில்", "டாக்டர்") and the word starts mid-consonant.
    #   pause      the silence left AFTER the last voiced sample. This is the
    #              inter-clause pause the caller actually hears. 0.08s reads as
    #              one continuous sentence; raise it toward 0.3s if the speech
    #              feels rushed to an elderly caller.
    clause_lead_seconds: float = float(os.getenv("TTS_CLAUSE_LEAD_SECONDS", "0.04"))
    clause_pause_seconds: float = float(os.getenv("TTS_CLAUSE_PAUSE_SECONDS", "0.08"))

    # Frames quieter than this count as silence. Edge's padding is true digital
    # silence, so almost any threshold separates it from speech; this one is
    # low enough that a soft Tamil word-final vowel is never mistaken for it.
    silence_threshold: float = float(os.getenv("TTS_SILENCE_THRESHOLD", "0.01"))

    def __post_init__(self) -> None:
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported TTS_LANGUAGE: {self.language}")
        if self.max_concurrency <= 0:
            raise ValueError("TTS_MAX_CONCURRENCY must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("TTS_TIMEOUT_SECONDS must be positive")
        # edge rejects anything that is not exactly +N%/-N%, and it does so
        # mid-call rather than at startup, so the shape is checked here.
        if not re.fullmatch(r"[+-]\d+%", self.rate):
            raise ValueError(f"TTS_RATE must look like '+15%' or '-10%', got {self.rate!r}")
        if self.engine not in {"edge", "svara"}:
            raise ValueError("TTS_ENGINE must be either 'edge' or 'svara'")


@dataclass(frozen=True)
class SecuritySettings:
    """Access control for the audio WebSocket - BACKEND_COMPLETION.md Sec4.

    No auth existed at all before this: anyone who could reach the port could
    open a session. A shared-secret token is the minimum viable gate for a
    single-tenant v1 deployment; swap for real per-caller auth (SIP trunk
    identity, a signed browser session token, ...) once there is more than one
    trusted caller of this service. Blank token means auth is OFF, matching
    this repo's dev-friendly-by-default env-var pattern (see TTS_VOICE_REFERENCE_PATH)
    - do not deploy to a reachable network with this left blank.
    """

    ws_auth_token: str = os.getenv("AUDIO_WS_AUTH_TOKEN", "")

    # Browser origins allowed to call the /api/* routes. The React dashboard is
    # served from its OWN origin (a Vite dev server, or a static host in
    # production), so without this its fetch of /api/health and /api/calls is
    # blocked by the browser and the whole dashboard reads as "backend down".
    # The /console page is exempt by construction - it is served BY this app,
    # so it is same-origin and never involves CORS.
    #
    # Comma-separated, e.g. "http://localhost:5173,https://aruvi.example.com".
    # "*" is accepted for a closed network but is NOT the default: it would let
    # any page on the internet a staff browser happens to open read the call
    # log, which is patient data. Blank means no cross-origin caller is allowed
    # at all, which is correct for a console-only deployment.
    #
    # This does NOT gate /ws/audio. WebSocket handshakes are exempt from CORS
    # in every browser, so the socket's gate is AUDIO_WS_AUTH_TOKEN above -
    # setting one without the other leaves the audio path open.
    cors_allow_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
        if origin.strip()
    )


@dataclass(frozen=True)
class PersistenceSettings:
    """Config for the call-event/history store - BACKEND_COMPLETION.md Sec3.6.

    SQLite (a single file, no server to run) is enough for v1's single-process
    deployment; swap for a real DB once concurrency (Sec3.5) moves this out of
    one process.
    """

    db_path: Path = Path(os.getenv("CALL_EVENTS_DB_PATH", str(_REPO_ROOT / "call_events.db")))

    # Fernet key (base64-encoded, from Fernet.generate_key()) for encrypting
    # event payloads at rest - BACKEND_COMPLETION.md Sec4 flags no
    # encryption-at-rest story for the ledger/call-recording data; this is
    # what makes that story real once a real key is set. Blank means
    # encryption is OFF (plaintext payloads), matching this repo's
    # dev-friendly-by-default env-var pattern (see TTS_VOICE_REFERENCE_PATH /
    # AUDIO_WS_AUTH_TOKEN) - do not deploy to a reachable network with this
    # left blank once real patient PII/PHI is flowing through the store.
    encryption_key: str = os.getenv("CALL_EVENTS_ENCRYPTION_KEY", "")
