# AICA-aruvi — handoff

`pytest -q` is green at **255 passed** in ~15s. Start with `./run.sh`.

**Current measured state** (LLM_TEST_RESULTS.txt Part 11, all at
`LLM_TEMPERATURE=0` except the socket numbers, which run the 0.3 product
default): `safety_eval` **0 violations / 9 turns**, `register_eval` 13/14 turns
and 2/7 calls, first audio **0.80 s** warm and **1.86 s** on the first call of
a fresh server, greeting 0.34 s.

The LLM half of the system has its own document: **[LLM_STACK.md](LLM_STACK.md)** —
model, prompt architecture, every measured number, and the approaches that were
built and rejected. Read that before touching the prompt. This file is the
short operational handoff.

---

## 0. HOW TO WORK ON THIS REPO

Each of these is a scar, not a style preference.

1. **Measure before you change anything.** Every claim in this file has a
   number behind it. A previous handoff stated three numbers confidently and
   all three were false, including the one it named as top priority.
2. **Exemplars are the specification, prose is commentary.** A rule written
   directly at a behaviour has made things worse every time it has been tried
   on this 4B model. A rule DEMONSTRATED in `golden/flow_exemplars.json` has
   held every time. See LLM_STACK.md §6 — including the case where an exemplar
   taught the wrong thing *by omission*.
3. **Run evals at `LLM_TEMPERATURE=0`.** At the 0.3 product default the same
   prompt scored 9/14, 13/14 and 9/14 in one sitting — noise wider than any
   change worth measuring.
4. **A test that cannot fail is worth nothing.** Every guard in this repo was
   verified by reintroducing its bug and confirming the test caught it. Two
   guards written this session initially passed with the code deleted; both
   were rewritten to drive the real call path. Do this for anything you add.
5. **Don't ship what measures worse.** English mirroring was built twice and
   reverted twice (LLM_STACK.md §9). That is the correct outcome, not a
   failure.
6. **Redirect stdout AND stderr to a file when you start the server.** Several
   real bugs were found only in that log, and a hidden window has none.

---

## 1. WHAT THIS IS

A real-time Tamil/English hospital phone agent. A caller speaks; the agent
listens, understands, takes the request down, and hands off to the desk.

**It talks, it does not transact.** There is deliberately no tool layer — see
LLM_STACK.md §5 for the measurement behind that. It cannot book, cancel,
dispatch or look anything up, and it says so honestly at the end of a call
rather than at the start.

---

## 2. RUNNING IT

```bash
.venv/Scripts/python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 \
  > logs/server.log 2>&1
```

Then `http://localhost:8000/console`. `GET /api/health` reports which of the
four components actually came up — a server that answers requests is not
evidence that ASR, the LLM or TTS is usable.

**The server caches prompts and code at startup.** Restart it after editing
`golden/runtime_core.txt`, `golden/flow_exemplars.json`, or any `backend/*.py`.

### Configuration lives in `.env`, and only in `.env`

Every knob this system has is in **`.env.example`**, commented at its shipped
default with the measurement behind it. Copy it to `.env` and uncomment what
you need — nothing here should require a code edit.

Both halves read it: `backend/__init__.py` loads it for the Python process
(from the package root, so it is in place before any module-level `os.getenv`
runs), and `run.sh` parses it for the shell, so `BACKEND_PORT`, `LLM_MODEL`,
`LLM_NUM_GPU` and the `OLLAMA_*` vars reach the model build and the Ollama
process too. A variable already exported in your shell still wins.

Four tests in `backend/test_env_coverage.py` keep that promise honest: every
`os.getenv` in the package and every `${VAR:-...}` in `run.sh` must appear in
`.env.example`, nothing may be documented that no longer exists, and a key
in your `.env` that matches no setting fails rather than being silently
ignored. Add a knob, document it, or the suite goes red.

Knobs that need a rebuild rather than a restart: `LLM_NUM_CTX`, `LLM_NUM_GPU`,
`LLM_TEMPERATURE` — they are baked into the Ollama model by
`python -m backend.scripts.setup_model`.

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*uvicorn*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

---

## 3. THE MACHINE — measured, do not re-derive

RTX 2050, **4096 MiB VRAM**. This is the binding constraint on everything.

| | |
|---|---|
| Ollama, `aruvi-base` | 3.1 GB resident, **100% GPU** |
| VRAM free with it loaded | **~500 MiB** (at `num_ctx 6144`) |
| `torch` | `2.13.0+cpu` — a CPU-only wheel, no CUDA kernels |
| IndicConformer checkpoint | 499 MB |

**The LLM now runs fully on the GPU. This reverses the old rule, and the
reversal is a measurement — read the whole of this before touching it.**

`PARAMETER num_gpu 99` was forbidden here for most of this project's life,
because it measured **114 seconds** on a real turn. That measurement was
correct. What it never recorded was its *precondition*: ~1.2 GB of the card was
being held by desktop apps (WhatsApp, Phone Link, CrossDeviceResume, Widgets
and their WebView2 GPU processes), leaving 439 MiB free. Under that pressure
the weights fit and the weights **plus** KV cache did not, and Windows WDDM
does not fail that allocation — it silently spills to system RAM over PCIe.

Close those apps and there are 3764 MiB free. Re-measured against real
`PromptBuilder` output (3703 tokens), temperature 0, identical reply text:

| config | split | generation |
|---|---|---|
| `num_ctx 8192`, no `num_gpu` | 33% CPU / 67% GPU | 12.9 tok/s |
| `num_ctx 5120`, no `num_gpu` | 29% CPU / 71% GPU | 14.7 tok/s |
| **`num_ctx 6144` + `num_gpu 99`** | **100% GPU** | **31.3 tok/s** |

**2.2x — the largest latency win measured on this project.** See
LLM_TEST_RESULTS.txt Part 10.

- **`num_gpu` is now an `.env` knob (`LLM_NUM_GPU`), blank by default, and the
  2.2x above is what you turn ON.** The measurement stands; hard-coding it does
  not. Current llama.cpp keeps a ~1 GiB free-VRAM target and normally recovers
  from missing it by offloading fewer layers — an explicit `num_gpu` removes
  that recovery. Measured on this box with an ordinary desktop running:

      projected to use 2915 MiB vs 3344 MiB of free device memory
      cannot meet free memory target of 1024 MiB, reduce by 595 MiB
      failed to fit params: n_gpu_layers already set by user to 99, abort

  That is `HTTP 500` from Ollama on **every turn** — the model never loads at
  all — not a slow turn. Reproduced twice. Blank, the same machine loads at
  22%/78% CPU/GPU and keeps answering. So: set `LLM_NUM_GPU=99` and rebuild
  (`python -m backend.scripts.setup_model`) on a machine whose card is
  genuinely free, confirm with `ollama ps`, and leave it blank on a shared one.
- **The VRAM precondition is not optional and not self-enforcing.** At
  `num_ctx 6144` full offload leaves ~500 MiB free. `num_ctx 8192` leaves 339
  MiB, too little for the browser a tester is about to open. If something takes
  the card, you get either the silent 114s spill or the hard 500 above.
- **There is now ONE build path**, and there used to be two that disagreed.
  `run.sh` and the manual route both go through
  `backend/scripts/setup_model.py`, which reads `LLM_NUM_CTX` / `LLM_NUM_GPU`
  out of `.env`. The script previously carried its own `NUM_CTX = 8192` while
  the Modelfile and `LLM_NUM_CTX` both said 6144 — and the script is what runs
  `ollama create`, so the model Ollama actually served had a window nothing in
  the repo claimed. The existing Modelfile/settings test passed the whole time,
  because it compared the two that agreed.
  `test_the_script_that_builds_the_model_agrees_with_the_modelfile` closes it.
- Benchmark only against real `PromptBuilder` output; a short prompt will lie
  to you. That part of the old warning was always right.
- The ASR still runs on CPU: the installed torch has no CUDA support, and there
  is not room for a 499 MB model beside the LLM anyway.

### Ollama environment — persisted user env vars, for the SERVER process

| Var | Value | Why |
|---|---|---|
| `OLLAMA_FLASH_ATTENTION` | `1` | part of what buys the 67% GPU share |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | same |
| `OLLAMA_KEEP_ALIVE` | `-1` | **the model used to unload after 5 min idle**, and the next call paid a full 3.6 GB reload — measured as a 14.9s first clause against a normal 1.3–2.6s |

**The Ollama tray app must be restarted after changing any of these** or it
keeps the old environment. Verify with `ollama ps`.

### Never write `localhost` in this repo

Measured: the identical `/api/chat` request takes **2.10s** via `localhost` and
**0.05s** via `127.0.0.1`. Name resolution tries an address the server is not
listening on and waits out a timeout first. Ollama never sees it, so it is
invisible in every server-side metric.

---

## 4. WHERE THE LATENCY IS — measured end to end over the real socket

**Every number here is a function of the CPU/GPU split, so read §3 first and
check `ollama ps` before comparing anything to it.** Re-measured over the real
socket at the shipping default (`LLM_NUM_GPU` blank → 22%/78% CPU/GPU):

| | 22%/78% split (default) | 100% GPU (`LLM_NUM_GPU=99`, free card) |
|---|---|---|
| Greeting, fully delivered | **0.05s** (TTS cache) | 0.05s |
| First audio, turns 2+ | **4.1 – 4.9s** | ~2.0s |
| First audio, turn 1 of a call | **~24s** | ~10s |
| Generation rate (`bench_ctx`) | **14.8 tok/s** | **34.4 tok/s** |
| Whole prompt+reply (`bench_ctx`) | **4.57s** | **1.98s** |

The 2.3x between the columns is the offload, and it is the only lever of that
size on this box. Everything else measured here is within noise of it.

**The server itself adds essentially nothing.** A warm turn's 4.1–4.9s to first
audio sits against `bench_ctx`'s 4.57s for the same prompt straight at Ollama —
VAD, ASR, clause chunking, TTS and the socket are all inside the rounding.
TTS per clause logs `0.00s synth` on any cached line.

**Turn 1 of a call is dominated by COLD PROMPT EVAL, not by model loading.**
`_system_prompt_for()`'s docstring has the mechanism: Ollama's prefix cache
gives 36 ms on an identical prefix and **28,895 ms** when a word near the top
changes. The first caller turn is exactly when that happens — the intent router
picks a playbook and swaps it into the prompt. The `prewarm` fired at
`call_started` covers **84.6%** of the resulting prompt (8782 of 10375 chars);
the flow-specific tail is re-evaluated. Shortening that tail, or moving it
after the stable block, is the remaining structural win — and it is a prompt
change, so read LLM_STACK.md §6 before attempting it.

Turn one of a freshly *started server* additionally pays the model load into
VRAM. `OLLAMA_KEEP_ALIVE=-1` stops that recurring mid-session.

---

## 5. THINGS THAT WILL BITE YOU

- **The ASR is Tamil-only.** English hospital words come back transliterated
  (`appointment` → `அப்பாயின்மென்ட்`) and dictated numbers come back as number
  *words*, in Tamil **or** English (`நீன் ஏஐட் போர் ஜெரோ...` = 9840...).
  `backend/transcript_norm.py` rewrites both over the hospital's closed
  vocabulary. **A new trigger word needs adding there, and to the intent router
  in both scripts.** Unknown words pass through untouched by design — fuzzy
  matching over Tamil script would eventually mangle a real Tamil word, which
  is worse than leaving one English word transliterated.
- **Two debounces sit on top of TEN VAD, and they are load-bearing.** One
  flagged 16 ms hop is not a turn. Measured over the 87 real captured turns in
  `call_events.db`, **46 transcribed to 3 characters or fewer and 34 to the
  EMPTY STRING** - 39% of everything the VAD opened held no speech at all, and
  each of those still ran the ASR and was free to cancel the agent. So:
  `vad_start_frames` consecutive hops open a turn (candidate audio is kept, so
  nothing is clipped), and `vad_resume_frames` consecutive hops are needed to
  restart the endpoint countdown - resetting it on a single hop is what let
  background noise hold the microphone open indefinitely.
- **A turn always ends, unconditionally — that is the watchdog.** Reported
  live: the FIRST turn of a call endpointed normally and every turn after it
  listened until the 30 s cap. Cause: the endpoint countdown is restarted by
  any sustained run of VAD-flagged frames, and once the agent has spoken,
  residual echo plus the browser's automatic gain control produce exactly such
  runs out of an empty room — so the countdown never completed. The flag alone
  cannot tell echo from speech; loudness can. A flagged frame now only
  restarts the countdown if it is also LOUD, and `vad_quiet_endpoint_frames`
  (704 ms of nothing loud) ends the turn regardless. Quiet frames are still
  KEPT in the utterance — audio is never discarded, only the countdown is
  affected. Guarded by
  `test_a_turn_always_ends_even_if_the_vad_never_stops_flagging_speech`.
  The watchdog is deliberately **twice** `endpoint_silence_frames`: equal would
  make a quiet syllable mean the same thing as silence, which is the reverted
  behaviour that cut turns off mid-word. `settings.py` refuses to start if the
  two are set equal.
- **Loudness gates turn ONSET, and gates the endpoint COUNTDOWN — but it can
  never count as silence frame-for-frame.** `vad_onset_min_rms` and
  `vad_onset_snr` are read in TWO places, and this bullet used to claim one,
  which is wrong in a way that matters: the not-yet-in-speech branch of
  `process()` (so a television across the room cannot start a turn) *and* the
  in-speech branch, where only a LOUD flagged frame restarts the countdown —
  that is the watchdog above. Once a turn is open, a frame that is merely
  flagged is kept in the utterance and restarts nothing.
  Guarded from both sides:
  `test_a_quiet_syllable_can_never_end_a_turn_that_is_already_open` and
  `test_a_turn_ends_when_the_caller_stops_even_though_the_room_stays_loud`.
  **That split is the whole design, and it is a scar.** An energy gate applied
  to EVERY frame was tried and reverted: a quiet trailing syllable scored as
  "silence", the endpoint countdown ran on through the middle of a word, and
  turns came back as one-character transcripts (`ந`, `ப`, `க`). Loudness may
  refuse to START a turn; it must never be able to END one. Guarded by
  `test_a_quiet_syllable_can_never_end_a_turn_that_is_already_open`, verified
  to fail when the gate is moved into the in-speech branch.
  Measured, per 16 ms frame of real Tamil speech at full digital level:
  voiced p10 **931**, p50 **2624**, peak **9804** — so the 200 default is a
  backstop about 4.6x below the quietest speech, not a filter. The SNR term is
  the part that adapts, and the noise floor learns only from frames the VAD
  calls non-speech, so a talking caller can never raise the bar against
  themselves.
- **A Latin word hyphenated to a Tamil suffix is dropped by the TTS voice.**
  `Cardiology-ல` synthesises 0.28s of speech; `Cardiology ல` synthesises 1.10s.
  `tts.speakable()` handles it on the synthesis path only — the transcript
  keeps the hyphen. Do not "clean that up".
- **`TTS_RATE` defaults to `+10%`.** This is slightly slower than the earlier
  +15% setting for clarity. Measured on a real reply: +0% = 6.12s of audio,
  +10% = 5.57s, +15% = 5.33s, +25% = 4.92s. Edge padding is trimmed to an
  80 ms inter-clause pause (`TTS_CLAUSE_PAUSE_SECONDS=0.08`).
  Past ~+25% the Tamil starts to clip, and callers here are often elderly.
- **Edge TTS is a network call** and this box's link to it degrades
  intermittently. The greeting and any repeated line are cached and unaffected;
  novel clauses can be slow or clipped while the link is bad. Check
  `logs/server.log` before suspecting the code.
- **`backend/tools.py` is dead at runtime** — nothing in the runtime imports it.
  It is kept as the reference for restoring tools per-flow later, and its
  seeded records are still used by an exemplar-disjointness test.
- **OUT OF SCOPE:** live SIP. `backend/telephony.py` gets fixes that are
  one-line-identical to `main.py`'s but is not otherwise developed.

---

## 6. INVARIANTS — each has already cost a session

- The core prompt must **lead with the agent's job**, not with what it cannot
  do. Leading with the limitation made the agent answer a booking request with
  "book பண்ண முடியாது". Guarded.
- Exemplar facts must stay disjoint from **both** `tools.py`'s seeded records
  **and** `golden/flows/*.txt`. Two tests, two different failure modes (silent
  memorisation vs. a fake PASS) — do not merge them.
- An exemplar that never demonstrates **asking** for a slot will have the model
  fill that slot from the exemplar's own facts.
- Grounding deliberately does **not** count the system prompt as a source. The
  exemplars live there.
- TTS clause audio must be sent by a consumer **independent of clause arrival**.
  Draining inline reintroduces the stall.
- Barge-in must be gated on **sustained** speech, never on `speech_started`.
  One frame is a cough.
- `MAX_HISTORY_MESSAGES` bounds the transcript. Without it a long call
  overflows `num_ctx` and Ollama truncates from the front, taking the language
  rules with it.

---

## 6a. THE INTENT ROUTER — measured, and what a statistical one cost

The router picks WHICH of the 20 playbooks the model is shown. A miss degrades
to `info.general`, which contains no guidance for whatever was actually asked.

Measured on 46 realistic hospital turns across all 20 flows:

| | routed correctly |
|---|---|
| before | **29/46 (63%)** — 11 matched nothing, 6 hit the wrong flow |
| after | 46/46 on that set, **17/20 on held-out data** |

Two structural bugs, both now guarded:

- **`appointment.confirm` and `postprocedure.checkin` had no pattern at all.**
  Twenty playbooks were parsed and eighteen were reachable. "Is my appointment
  confirmed?" matched `appointment.book` and the agent tried to take a fresh
  booking. `test_every_playbook_can_actually_be_reached_by_the_router` now
  fails the moment a flow is added without a trigger — that failure is
  otherwise silent.
- **`charge` had no word boundary, so it matched inside DIScharge**, sending
  "discharge summary copy வேணும்" to billing instead of records.

**A derived statistical router was built and REJECTED.** Scoring a turn against
per-flow token profiles built automatically from the playbooks and the
exemplars' caller turns — the obvious answer to "stop hardcoding patterns" —
measured **52–57%, worse than the 63% regex it would have replaced**, at every
threshold tried. The cause is data volume: two or three exemplar caller turns
per flow, and playbook bodies that share most of their vocabulary. Same class
of failure as the fuzzy ASR normaliser in §6c, and rejected for the same
reason. The prototype is not in the repo.

**Held-out validation is what caught a regression.** Tuning patterns against
the 46-turn set scored 100%, which is meaningless — those turns were what the
patterns were written against. Re-running against the exemplars' own opening
turns (never looked at while writing patterns) exposed a new
`ரெண்டு தடவை` ("twice") trigger on the angry-escalation flow hijacking
*"charged twice"*, an ordinary billing dispute. Always measure on turns you
did not write.

The three still unrouted are exemplar openers for OUTBOUND calls
("ஆமாம் ஞாபகம் இருக்கு" — "yes, I remember"). They are answers, not requests;
nothing in them names a flow, and the sticky-intent rule in `CallSession`
covers them.

---

## 6b. LATENCY IS AT THE HARDWARE FLOOR — measured, stop looking

Three separate attempts, all measured, all dead ends. Do not repeat them.

| Attempt | Result |
|---|---|
| `FIRST_CHUNK_MAX_CHARS` 32 -> 24 -> 18 | **No change.** 2.38s / 2.43s / 2.43s to first clause. The model's first sentence ends at a period (`கண்டிப்பா சார்.` = 15 chars) long before the 32-char cap applies, so the knob never fires. At 12 it finally moved (2.37 -> 1.89s) but only by cutting mid-phrase, dropping `சார்.` |
| openai SDK vs raw Ollama SSE | **Identical.** 9.0 vs 9.1 tok/s. The client adds nothing; rewriting `llm.py` to use raw httpx would buy zero. |
| `OLLAMA_NUM_PARALLEL` | **Already 1.** The server log confirms one slot with the full `n_ctx 8192`. Flash-attention, q8_0 and keep-alive are all applied. |

Where a turn actually goes, measured on the production path:

    time to first TOKEN     0.42s (warm)  /  1.09s (turn 1, flow re-eval)
    generating first clause 1.35s
    -> first clause         1.79s (warm)  /  2.42s (turn 1)

Generation streams at **9 tok/s**, against 14.5 tok/s for the same prompt
non-streamed. That gap is llama.cpp flushing per token, not our code. It is the
floor on this hardware. Real fixes are more VRAM or a smaller quantisation
(measure `register_eval` before and after).

The one lever that DID work is `TTS_RATE` - see below.

---

## 6c. GENERALISING THE ASR NORMALISER — prototyped, MEASURED, rejected

`backend/transcript_norm.py` uses a lookup table, and the obvious objection is
that it is hardcoded. A general replacement was built and measured:

  1. derive the vocabulary automatically from the prompt files (1570 Latin
     words - so a new department in the prompt would work for free),
  2. transliterate the Tamil token with a script-level table,
  3. fold the distinctions Tamil cannot express (p/b, k/g, t/d), fuzzy match.

**It is worse than the table on both axes.** Against the measured ASR outputs:

| variant | recovered | corrupted real Tamil |
|---|---|---|
| fuzzy, vowels dropped | 6/14 | **8/12** - `வணக்கம்`→"income", `மருந்து`→"rent" |
| fuzzy, vowel positions kept | 8/14 | **3/12** - `எனக்கு`→"intake", `சொல்லுங்க`→"silence" |
| exact canonical match only | 5/14 | **2/12** - `சொல்லுங்க`→"silence", `டெஸ்ட்`→"dust" |

Even the strictest variant corrupts `சொல்லுங்க` ("tell me"), which appears in
almost every caller turn. Turning a caller's real word into the wrong English
word is far worse than leaving one English word transliterated - the table's
whole design principle.

If you retry this: the failure is that Tamil script is lossy for English
(no b/g/d, unreliable vowels), so short tokens collide. A better direction is
FORWARD transliteration - generate expected Tamil forms from the derived
English vocabulary and accept only exact hits - which cannot invent a match for
a word that no English source produced. Not attempted.

### The retry, in that direction, and what shipped

`backend/scripts/build_asr_lexicon.py`. It does not transliterate by rule at
all - it ROUND-TRIPS: speak the English word with the agent's own Tamil voice,
transcribe it with the caller's own ASR, record what came back. A rule guesses
`appointment` as `அப்பொஇன்ட்மென்ட்`; the model actually emits
`அப்பாயின்மென்ட்`, and only the model knows that.

Nothing about the vocabulary is hand-maintained. It is every Latin word in
`golden/`, so a department added to the prompt is covered by the next build.

The hazard is that `golden/` also writes plenty of Tamil in Latin letters
(`aamaam`, `aduttha`, `appadi`, `aiyo`), and round-tripping those would teach
the normaliser to rewrite real Tamil into Latin - the exact corruption above.
Two derived screens stop it, no word lists:

1. **The code-mix register.** This prompt teaches Tamil-English code-mix, so a
   real English word appears as a Latin island inside a Tamil-script sentence
   (`எனக்கு appointment book பண்ணணும்`), while romanised Tamil appears in lines
   that are romanised throughout. Requiring Tamil script on the SAME LINE keeps
   the first and drops the second. Measured on 16 hand-picked words of each
   kind: **romanised Tamil admitted 1/16, English kept 15/16.**
2. **A similarity backstop**, for the one that got through. `அண்ணா` (elder
   brother) scores 0.80 against `பண்ணா`; anything that close to a word the
   agent writes in real Tamil is rejected. Deliberately strict - losing an
   English word costs coverage, admitting a Tamil one costs correctness.

Plus the original screens: exact collision with real Tamil, ambiguity (two
English words producing one Tamil form), and a minimum length.

The result is merged **under** the hand table, so a generated entry can only
add coverage and never overwrite a measured one, and a missing or malformed
`golden/asr_lexicon.json` degrades to exactly the hand table. Matching is
still exact and whole-word: a form no English source produced cannot be
matched, which is why the `சொல்லுங்க`->"silence" class of failure is
structurally impossible here rather than merely unlikely.

Rebuild after editing the prompt:

```bash
.venv/Scripts/python.exe -m backend.scripts.build_asr_lexicon
```

---

## 7. OPEN

0. ~~**The three clinical refusals collapse under pressure**~~ — **closed, by
   exemplar.** `safety_eval` 3 violations → **0**. Each of the three exemplars
   now demonstrates a THIRD push met with the *same* refusal words, refusal
   first. Sec6's rule again: prose had asked for exactly this ordering for
   several sessions and the model followed the demonstration instead.
   LLM_TEST_RESULTS.txt Part 11.2.

   Two traps came with it, both guarded now. **Never write an exemplar's caller
   line from an eval turn** — it destroys what the eval measures *and* makes
   the model read the line aloud as its own turn
   (`test_no_exemplar_caller_line_restates_an_eval_scenarios_turn`). And
   **exemplar length is not free**: the same three edits pushed the worst-case
   prompt 155 tokens past `num_ctx`
   (`test_a_call_of_long_turns_never_overflows_num_ctx`), and a longer block
   moved an unrelated turn's register into Tamil script — a defect no single
   line in it owns (Part 11.4).

0b. **`safety_eval` shares turns with two of its own exemplars** — five
   overlaps predate the guard above and are pinned, not fixed. On two of its
   three cases it has been scoring recall alongside generalisation. Fixing it
   means a held-out scenario set, and it invalidates comparison with every
   safety number on record, so do it deliberately. **First job for the next
   session** (Part 11.3).

0c. **The context budget must be re-derived whenever `num_ctx` moves**, not
   only when a prompt grows. Measured this session: the shipped config was
   **1058 tokens over** `num_ctx 6144`, because Sec10.2 sized
   `MAX_HISTORY_MESSAGES` against 8192 and the window was later lowered for
   VRAM headroom. `conversation.py` now trims history by characters against a
   budget derived from `LLM_NUM_CTX` (which must match the Modelfile; a test
   checks). Part 11.5.

1. ~~**Two questions in one turn**~~ — **guarded.** Prose failed three times;
   the fix is deterministic and lives in `stream_utterance`. It is not a
   truncation of the reply: measured against the real recorded calls, a
   two-question turn arrives as two SEPARATE clauses, so the second is still
   unspoken when it closes and can simply be withheld. Later NON-question
   clauses are kept, because the closing handoff line usually follows the
   question and dropping the tail wholesale would lose it. `?` is the same test
   `register_eval` scores with, deliberately - one definition, so the guard and
   the eval cannot disagree. The turn recorded in history is what was SPOKEN,
   not what was generated, or the model believes it asked something the caller
   never heard.
2. **English mirroring** — see LLM_STACK.md §9. Attempted twice, reverted
   twice, with the measurements.
3. **TTS off the network** — still the right end state for both privacy and
   reliability. `facebook/mms-tts-tam` was downloaded and **rejected**: its
   vocabulary is 59 tokens with one Latin letter, so
   `"Cardiology-ல appointment book பண்ணணும்"` tokenises to `"a ல a  பண்ணணும்"`.
   It cannot speak code-mix. Do not retry it.
