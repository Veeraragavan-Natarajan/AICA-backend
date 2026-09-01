# AICA backend — Local Setup Guide

This covers running the **AICA backend** locally. The React admin dashboard that drives it
lives on the separate [`frontend-aica-ui`](../../tree/frontend-aica-ui) branch — check that
branch's own SETUP/README if you need the two running together for
*Simulation & Testing → Run simulation*.

## Architecture, in one line

```
Browser mic/keyboard --WebSocket--> FastAPI /ws/audio --> TEN VAD --> IndicConformer ASR (Tamil)
      --> Conversation Manager --> LLM (OpenAI-compatible, e.g. Ollama) --> clause-by-clause TTS
      --> agent reply text streams back to the browser transcript
```

The whole chain is wired up and runs end to end, voice in to voice out: verify it on
your own machine with `python -m backend.scripts.e2e_check` against a running server,
which synthesizes a caller's Tamil line, streams it in as 16 kHz PCM, and checks that a
transcript came back and the reply carried both text and audio.

Only `TTS_ENGINE=svara` remains a placeholder; the default `edge` engine is real (see
the TTS section under Component
recommendations below) — voice replies don't exist yet regardless of what you install.

## Setup

```bash
py -3.11 -m venv .venv && .venv/Scripts/Activate.ps1   # or py -3.10; source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
git clone --depth 1 https://github.com/AI4Bharat/NeMo.git NeMo_ai4bharat
pip install -e ./NeMo_ai4bharat --no-deps
cp .env.example .env
# add HF_TOKEN=hf_... to .env (see step below)
# install Ollama, then: ollama pull qwen2.5:7b   (or 3b/14b/32b — see sizing table below)
uvicorn backend.main:app --reload    # starts the backend on :8000
```

## Configuration — all of it is in `.env`

`.env.example` is the complete list of knobs: server host/port, log level and
format, CORS origins, the WebSocket auth token, every LLM/TTS/ASR/VAD/barge-in
parameter, the call-log path and its encryption key, and the `OLLAMA_*` vars
that must reach the Ollama process. Each is commented at its shipped default
with the measurement behind it. Copy the file, uncomment what you need; nothing
should require a code edit.

`run.sh` parses `.env` for the shell as well, so `BACKEND_PORT`, `LLM_MODEL`
and `LLM_NUM_GPU` reach the model build and the Ollama process — not just the
Python app. A variable already exported in your shell wins over the file.

Most knobs need only a server restart. `LLM_NUM_CTX`, `LLM_NUM_GPU` and
`LLM_TEMPERATURE` are baked into the Ollama model, so rerun
`python -m backend.scripts.setup_model` after changing those.

`backend/test_env_coverage.py` fails if a setting exists that `.env.example`
does not document, if `.env.example` documents one nothing reads, or if your
own `.env` sets a key that matches no setting — a typo'd key is otherwise
completely silent.

**What can't be scripted:** a Hugging Face token for the gated ASR model
(`ai4bharat/indicconformer_stt_ta_hybrid_ctc_rnnt_large`). It's tied to your personal
HF account and a license you have to accept yourself:
1. Visit the [model page](https://huggingface.co/ai4bharat/indicconformer_stt_ta_hybrid_ctc_rnnt_large)
   and accept its license.
2. Create a **read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens),
   and set it as `HF_TOKEN` in `.env`.

`ollama serve` needs to be running and reachable at `.env`'s `LLM_BASE_URL`, with the
model in `LLM_MODEL` already pulled (`ollama pull <tag>`).

`./run.sh` does all of the above in one command - see README.md. The rest of this file
is the manual walkthrough, for when you need to change something in the middle of it.

---

## Component recommendations

### The prompt is assembled per turn — do not send `main_prompt.txt` whole

`golden/main_prompt.txt` is a ~15k-token **specification**, not a runtime prompt.
Sending it whole as the system message produced two separately-reproduced failures:

| num_ctx | What happened |
|---|---|
| Ollama default (2048–4096) | Prompt silently truncated **before** section 2's language rules. Agent replied in fluent English and called `lookupPatient` with an invented mobile number while the real one sat in the ledger. |
| 32768 | Truncation fixed, but the KV cache is far too large for CPU/small-GPU — a *trivial* generation measured **222 s**, and the reply came back as Tamil script with no coherent meaning. |

So `backend/prompt_builder.py` assembles the prompt instead:

```
golden/runtime_core.txt        rules that apply to EVERY turn
  + ONE flow playbook          parsed out of main_prompt.txt section 8
  + that flow's exemplars      golden/flow_exemplars.json
  = ~2.5-3k tokens             (vs ~15k)
```

A deterministic keyword router (`detect_intent`) picks the flow with no extra LLM
round-trip — a voice turn cannot afford one. The intent is **sticky** per call, so a
follow-up turn that matches no trigger ("ஆமாம்", a phone number) does not drop the
playbook mid-flow. `main_prompt.txt` remains the single source of truth for the 20
playbooks; they are parsed from it, never duplicated.

**Always set `num_ctx` explicitly** in a Modelfile (see the repo-root `Modelfile`,
which pins 8192). Ollama's default is VRAM-derived and will silently truncate.

Verify register — Tamil/English ratio, wrong-script leakage, unspeakable symbols,
turn length, questions per turn, fabricated identifiers — on scenarios that appear
nowhere in `golden/`:

```bash
python -m backend.scripts.register_eval
```

This is a **floor**: clean means nothing is mechanically wrong, not that the Tamil
reads naturally. A native reader still has to read the transcript.

### LLM — Qwen3-4B-Instruct-2507

`qwen2.5:3b` does not clear the bar for this prompt even with the context bug fixed —
un-truncated, it produces Tamil script with no coherent meaning. That is a capacity
ceiling, not a configuration problem.

**Qwen3-4B-Instruct-2507** (Apache-2.0, non-thinking, 262144-token native context) is
the verified default. Build it with the pinned context:

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M
python -m backend.scripts.setup_model     # creates aruvi-base from the best base you
                                          # actually have installed, using LLM_NUM_CTX,
                                          # LLM_TEMPERATURE and LLM_NUM_GPU from .env
# .env already says LLM_MODEL=aruvi-base
```

This is the **only** build path — `run.sh` calls the same script. It used to be
one of two, and they disagreed: the script pinned its own `num_ctx 8192` while
the Modelfile and `LLM_NUM_CTX` both said 6144, so whichever had been run last
decided what Ollama really served. Rerun it after changing `LLM_NUM_CTX`,
`LLM_NUM_GPU` or `LLM_TEMPERATURE` — those are baked into the model, and a
restart alone will not pick them up.

> **Latency warning:** on CPU this measures ~3.6 tok/s, i.e. minutes per turn — fine
> for evals, unusable for a live voice call. GPU offload is required for real use.

### Older sizing notes — Qwen2.5-Instruct, not Llama 3.1

`backend/settings.py`/`BACKEND_COMPLETION.md` name Llama-3.1-8B-Instruct and
Qwen2.5-7B-Instruct as the two candidates the backend was designed around (it talks to
either through one OpenAI-compatible client — swappable via `LLM_BASE_URL`/`LLM_MODEL`
alone, no code changes). Between those two, **Qwen2.5-Instruct is the better fit here**:

- This is a Tamil/English code-mixed hospital call agent — Qwen2.5's tokenizer and
  training mix cover non-Latin scripts noticeably more broadly than Llama 3.1's, whose
  officially supported languages don't include Tamil at all.
- `backend/llm.py` streams plain chat completions; the agent has no tool layer, so a
  model does NOT need the "Tools" badge on
  [ollama.com/library](https://ollama.com/library). What it does need is Tamil
  fluency and instruction-following at 4B. Check a model's tag before relying
  on it here.

**Pick a size by hardware** (rule of thumb — VRAM needs scale roughly with parameter
count at a given quantization, and IndicConformer/TEN VAD also share whatever GPU you
have):

| Hardware | Model tag | Notes |
|---|---|---|
| CPU-only / <6GB VRAM | `qwen2.5:3b` | Fast, but weaker Tamil fluency and prone to degenerate repetition — expect a worse `register_eval` score. |
| 6–8GB VRAM | **`qwen2.5:7b`** (recommended default) | Best balance for this domain at MVP scale. |
| 12GB+ VRAM | `qwen2.5:14b` | Noticeably better instruction-following on the harder flows (emergency escalation, multi-step booking); higher per-turn latency. |
| 24GB+ VRAM / cloud | `qwen2.5:32b`, or move to vLLM/TGI for a served 7B–14B | Only worth it once you have concurrent calls to serve — `BACKEND_COMPLETION.md` §3.5 already flags Ollama/single-process as a later scaling limit, not an MVP one. |

**Don't just trust this table — validate it against your own prompt and flows:**

```bash
LLM_TEMPERATURE=0 python -m backend.scripts.register_eval
LLM_TEMPERATURE=0 python -m backend.scripts.safety_eval
```

`register_eval` scores the Tamil/English register on scenarios that appear NOWHERE in
`golden/`, so a model that merely memorised the twenty flows scores badly; `safety_eval`
checks the three clinical refusals hold under repeated pressure. Run both at
`LLM_TEMPERATURE=0` — at the 0.3 product default the same prompt scored 9/14, 13/14 and
9/14 in one sitting, noise wider than any change worth measuring.
Re-run them any time you switch models — it's the real signal for *this* prompt, not a
generic multilingual benchmark.

### Test console — `http://localhost:8000/console`

A single self-contained page served by the backend itself, so it shares the app's
origin and opens `/ws/audio` with no CORS and no second dev server:

```bash
uvicorn backend.main:app --reload
# then open http://localhost:8000/console
```

It shows live readiness for socket / conversation / voice / speech-to-text, and gives
you two ways in:

- **Typed turns** — uses the socket's `user_text` path, which skips only VAD/ASR and
  drives the identical conversation → LLM → TTS chain a spoken turn does. This
  works **without `HF_TOKEN` or the NeMo fork**, which is the part most likely to be
  missing on a fresh machine.
- **Microphone** — the full VAD → ASR path. Enabled only when ASR actually loaded.

One-click sample turns (booking, refill, report, billing, info, and the emergency
override) are lifted from the golden flows so you can exercise several intents without
composing Tamil by hand. Agent replies arrive as text clauses plus raw int16 PCM at
the rate announced in `agent_speaking_start`, scheduled back-to-back in the browser.

### TTS — a real female Tamil voice, with a privacy caveat

`TTS_ENGINE=edge` (the default) uses Microsoft's neural voices: **no API key, no model
download, runs on CPU**. Tamil defaults to **`ta-IN-PallaviNeural`** (female); override
with `TTS_VOICE` (e.g. `ta-IN-ValluvarNeural` for male). Output is 24 kHz mono int16.

> **This sends each reply's text to Microsoft to be synthesised.** That is fine for the
> fictional Aruvi data in `golden/`, and **not** fine for real patient speech — this is a
> hospital agent and the text is PHI-adjacent, the exact gap `BACKEND_COMPLETION.md` §4
> already flags. Before anything resembling production, swap in a self-hosted engine
> behind the same adapter interface; nothing else in the pipeline changes.

`TTS_ENGINE=svara` still selects the original placeholder, whose `load()` raises until a
real svara-TTS reference exists. A failed TTS load is **not** fatal — `speak()` degrades
to text-only `agent_clause` events, which is what the `ready` gate is for.

### Self-hosted TTS — the production path

`backend/tts.py`'s `load()` is a literal `raise NotImplementedError` placeholder — there
is no real TTS integration in this backend at all today. No voice output will ever come
back, no matter what you install; the agent's replies stream back as text
(`agent_clause` events) instead, which the frontend already renders. When you're ready
to build the real thing:

- **[AI4Bharat Indic Parler-TTS](https://huggingface.co/ai4bharat/indic-parler-tts)**
  (recommended first try) — open-source, self-hosted, same organization as your ASR
  model so it already targets Tamil script well, and keeps caller audio on your own
  infrastructure. That last point matters: this is a hospital agent handling
  PHI-adjacent data, and routing audio through a third-party cloud TTS reopens the
  exact compliance gap `BACKEND_COMPLETION.md` §4 already flags for the project.
- **Coqui XTTS-v2** — genuine few-shot voice cloning from a short reference clip and
  strong tooling, but Tamil isn't in its officially supported language list. Worth an
  empirical try, no guarantee of quality.
- A cloud API (e.g. ElevenLabs) is the fastest to bolt on and has real Tamil + cloning
  support, but sends caller audio off-device — fine for throwaway local experiments,
  not for anything closer to production given the PHI concern above.

### ASR / VAD — already the right choice, no action needed

IndicConformer (ASR) and TEN VAD are already well-suited, tested, and hard-wired for
this exact use case (Tamil conversational turn-taking) — there's nothing to swap here
for the MVP.

---

## Known limitations right now

- **No voice output** — TTS is a placeholder (see above).
- **No real phone line** — `backend/telephony.py` is a code-complete, unit-tested
  adapter for Twilio-Media-Streams-shaped input, but has never been wired to an actual
  SIP trunk/DID; only the browser WebSocket (`/ws/audio`) is a live transport today.
- **Single-process concurrency** — one global ASR/TTS semaphore pool per
  `AudioSettings`/`TtsSettings`; fine for one person testing, not load-tested for many
  simultaneous callers.

## Troubleshooting

- **`ready` event's `asr_ready` is false** → check `HF_TOKEN`, that the NeMo fork
  installed cleanly, and the backend's startup logs for an ASR load error.
- **`conversation_ready` is false, or nothing replies to voice/typed text** → LLM
  unreachable → confirm Ollama is running (`ollama list`) and that `.env`'s
  `LLM_BASE_URL`/`LLM_MODEL` match what you actually pulled.
- **A connected client can't reach the backend at all** → confirm `uvicorn` is running
  on the port your client points at (default `ws://localhost:8000/ws/audio`).

## GPU settings (measured on a 4GB RTX 2050)

Two environment variables must be set for the Ollama **server** process, not for
the backend. They are already persisted for the current user; set them again on
a new machine:

```
setx OLLAMA_FLASH_ATTENTION 1
setx OLLAMA_KV_CACHE_TYPE q8_0
```

Then restart Ollama (quit the tray app and reopen it) so the server picks them
up, and check with `ollama ps` that `SIZE` dropped.

Why: at `num_ctx 8192` an f16 KV cache costs ~1.2GB on top of ~2.5GB of weights,
which does not fit in 4GB, so Ollama silently keeps ~40% of the model on the CPU.
Quantising the KV cache to q8_0 halves it **without losing any context**:

| | generation |
|---|---|
| default (f16 KV) | 16.6 tok/s, 58% of the model in VRAM |
| flash attention + q8_0 KV | 27.2 tok/s, 67% in VRAM |

### Full GPU offload — `LLM_NUM_GPU`, and why it is off by default

This section used to read "**Do not add `PARAMETER num_gpu 99`**", on the
strength of a turn that measured **114 seconds**. That measurement was real but
its precondition was not recorded: ~1.2GB of the card was held by desktop apps
at the time, so the weights fit and the weights **plus** the KV cache did not,
and Windows WDDM spills the difference to system RAM over PCIe rather than
failing. With the card actually free, full offload measures **2.2x** — the
largest latency win on this project:

| | generation | prompt+reply |
|---|---|---|
| auto-fit (default, 22%/78% CPU/GPU) | 14.8 tok/s | 4.57s |
| `LLM_NUM_GPU=99` (100% GPU) | **34.4 tok/s** | **1.98s** |

It is still **off by default**, for a second reason found since. Current
llama.cpp keeps a ~1 GiB free-VRAM target and normally recovers from missing it
by offloading fewer layers; an explicit `num_gpu` removes that recovery:

```
projected to use 2915 MiB vs 3344 MiB of free device memory
cannot meet free memory target of 1024 MiB, reduce by 595 MiB
failed to fit params: n_gpu_layers already set by user to 99, abort
```

The model then never loads and **every turn returns HTTP 500** — not a slow
turn. Blank, the same machine runs at a CPU/GPU split and keeps answering.

So: set `LLM_NUM_GPU=99` in `.env` and rerun `setup_model` on a machine whose
card is genuinely free (close the Electron apps first), confirm `ollama ps`
reads `100% GPU`, and blank it again the moment it 500s. Tune only against a
realistic prompt — `backend/scripts/bench_ctx.py` assembles the real one; a
short prompt measured 34.5 vs 16.6 tok/s while the real one took 114 seconds.
