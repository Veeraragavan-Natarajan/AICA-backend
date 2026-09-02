#!/usr/bin/env bash
#
# Start AICA. One command, from anywhere:
#
#     ./run.sh
#
# On a machine that has Python 3.10/3.11 and Ollama installed, this is the only
# command needed. It finds a supported Python, builds .venv if missing,
# installs requirements only when they are actually missing, seeds .env,
# STARTS Ollama if it is not already up (with the three environment settings
# that are worth 2.2x, which only apply if they are set before Ollama starts),
# pulls the base model and builds aruvi-base from the Modelfile, preloads it so
# the first caller never pays the cold-start cost, starts exactly one API
# worker, waits until every component and the fixed-phrase TTS cache are ready,
# and prints the console URL.
# Ctrl+C stops it cleanly.
#
# It does NOT install Python or Ollama - those are system installs and it says
# where to get them instead.
#
#   BACKEND_PORT=9000 ./run.sh     serve on another port
#   BACKEND_HOST=0.0.0.0 ./run.sh  bind elsewhere (read the warning it prints)
#   REINSTALL=1 ./run.sh           force a dependency reinstall
#   RELOAD=1 ./run.sh              uvicorn --reload (development)
#   SKIP_OLLAMA=1 ./run.sh         don't touch Ollama (it is remote, or managed)
#
set -Eeuo pipefail

# Works no matter where it is invoked from, including through a symlink.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="$ROOT_DIR/.venv"
LOG_DIR="$ROOT_DIR/logs"

# ------------------------------------------------------------------ .env ----
# Seeded and loaded HERE, before anything below reads a variable out of it.
# backend/__init__.py loads .env for the PYTHON process, but this shell needs
# it too: BACKEND_PORT, BACKEND_HOST, LLM_MODEL and OLLAMA_URL are all read by
# this script, and a .env only the app could see meant setting LLM_MODEL there
# while run.sh went on pulling and building the default one.
#
# Parsed line by line rather than sourced: a .env is data, and sourcing it
# executes it. An unquoted value with a space in it - a comma-separated
# CORS_ALLOW_ORIGINS is the realistic one - would run its second word as a
# command. A variable already exported into this shell still wins over the
# file, so `BACKEND_PORT=9000 ./run.sh` keeps working, matching dotenv.
if [[ ! -f "$ROOT_DIR/.env" && -f "$ROOT_DIR/.env.example" ]]; then
  cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
  printf '\033[36m==>\033[0m %s\n' "Created .env from .env.example"
  printf '\033[33m !\033[0m %s\n' "Set HF_TOKEN in .env for the microphone (the ASR model is gated)." >&2
  printf '\033[33m !\033[0m %s\n' "  Without it the server still runs and /console's typed turns work." >&2
fi
if [[ -f "$ROOT_DIR/.env" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"                                    # CRLF-safe
    [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
    [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    env_key="${BASH_REMATCH[2]}"
    env_value="${BASH_REMATCH[3]}"
    env_value="${env_value%"${env_value##*[![:space:]]}"}"  # trailing space
    case "$env_value" in
      \"*\"|\'*\') env_value="${env_value:1:${#env_value}-2}" ;;
    esac
    [[ -n "${!env_key+set}" ]] || export "$env_key=$env_value"
  done < "$ROOT_DIR/.env"
  unset env_key env_value line
fi

BACKEND_PORT="${BACKEND_PORT:-8000}"
# logs/server.log for the default port, because that path is what HANDOFF.md
# and every debugging note refer to. A second instance on another port gets its
# own file rather than overwriting the first one's - two servers clobbering one
# log is how you end up reading the wrong process's output.
if [[ "$BACKEND_PORT" == "8000" ]]; then
  LOG_FILE="$LOG_DIR/server.log"
else
  LOG_FILE="$LOG_DIR/server-$BACKEND_PORT.log"
fi
# 127.0.0.1, never "localhost" - measured, the identical /api/chat request takes
# 2.10s via localhost and 0.05s via 127.0.0.1 on this stack.
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
MODEL_NAME="${LLM_MODEL:-aruvi-base}"
# Read out of the Modelfile rather than restated here: two copies of a model
# name is how they drift.
BASE_MODEL="$(sed -n 's/^FROM[[:space:]]\{1,\}//p' "$ROOT_DIR/Modelfile" | head -1)"

mkdir -p "$LOG_DIR"

say()  { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m !\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m x\033[0m %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------- python ----
# IndicConformer's NeMo build has no wheels above 3.11 - numba and editdistance
# try to compile from source and fail - so the range is pinned rather than
# "3.10 or newer".
SUPPORTED='import sys; raise SystemExit(not ((3,10) <= sys.version_info[:2] <= (3,11)))'
PYTHON_CMD=()

find_python() {
  local candidate version
  for candidate in python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c "$SUPPORTED" >/dev/null 2>&1; then
      PYTHON_CMD=("$candidate"); return
    fi
  done
  for version in 3.11 3.10; do
    if command -v py >/dev/null 2>&1 && py "-$version" -c "$SUPPORTED" >/dev/null 2>&1; then
      PYTHON_CMD=(py "-$version"); return
    fi
  done
  die "Python 3.10 or 3.11 is required (NeMo does not support 3.12+).
     Install from https://www.python.org/downloads/ with 'Add Python to PATH'."
}

if [[ ! -d "$VENV_DIR" ]]; then
  find_python
  say "Creating .venv with ${PYTHON_CMD[*]}"
  "${PYTHON_CMD[@]}" -m venv "$VENV_DIR"
fi

# Windows venvs put the interpreter in Scripts/, which is what Git Bash sees.
if   [[ -x "$VENV_DIR/bin/python"        ]]; then PY="$VENV_DIR/bin/python"
elif [[ -x "$VENV_DIR/Scripts/python.exe" ]]; then PY="$VENV_DIR/Scripts/python.exe"
else die ".venv is incomplete. Delete it and run again."
fi

"$PY" -c "$SUPPORTED" >/dev/null 2>&1 \
  || die ".venv was built with an unsupported Python (NeMo needs 3.10 or 3.11).
     Delete .venv and run again."

# ----------------------------------------------------------- dependencies ----
# Checked by import, not reinstalled every run: a full pip pass over this
# requirements file takes minutes and the point of this script is that it can
# be run casually.
deps_present() {
  "$PY" - <<'EOF' >/dev/null 2>&1
import importlib.util as u
raise SystemExit(any(u.find_spec(m) is None for m in ("fastapi", "uvicorn", "openai", "edge_tts", "soundfile", "ten_vad")))
EOF
}

if [[ -n "${REINSTALL:-}" ]] || ! deps_present; then
  say "Installing requirements (first run takes a while)"
  "$PY" -m pip install --upgrade pip --quiet
  "$PY" -m pip install -r "$ROOT_DIR/requirements.txt"
else
  say "Dependencies present"
fi

# IndicConformer needs the AI4Bharat NeMo fork; stock nemo-toolkit cannot load
# its multilingual tokenizer. --no-deps skips Windows-incompatible extras.
if ! "$PY" -c "import nemo.collections.asr" >/dev/null 2>&1; then
  if [[ -d "$ROOT_DIR/NeMo_ai4bharat" ]]; then
    say "Installing the AI4Bharat NeMo fork"
    "$PY" -m pip install -e "$ROOT_DIR/NeMo_ai4bharat" --no-deps
  else
    warn "NeMo_ai4bharat/ is missing - the microphone will be unavailable."
    warn "  git clone --depth 1 https://github.com/AI4Bharat/NeMo.git NeMo_ai4bharat"
    warn "  Typed turns in the console still exercise the full LLM -> TTS chain."
  fi
fi

# ---------------------------------------------------------------- ollama ----
# The agent is useless without the LLM, so all of this happens before the
# server starts rather than surfacing later as a failed turn.
ollama_up() { curl -fsS -m 5 "$OLLAMA_URL/api/tags" >/dev/null 2>&1; }

if [[ -n "${SKIP_OLLAMA:-}" ]]; then
  # Nothing below runs, including `ollama create`: under SKIP_OLLAMA the server
  # may well be on another machine, and the local CLI would build the model on
  # THIS one, where nothing would ever read it.
  if ollama_up; then
    say "Using the Ollama at $OLLAMA_URL as it is (SKIP_OLLAMA)"
  else
    warn "SKIP_OLLAMA is set and nothing is answering at $OLLAMA_URL."
  fi
elif ! ollama_up && command -v ollama >/dev/null 2>&1; then
  # THESE THREE MUST BE EXPORTED BEFORE OLLAMA STARTS. Ollama reads them from
  # the environment of its own server process, which is why this script used to
  # only be able to warn about them. Starting Ollama ourselves is what makes
  # them applicable on a machine nobody has configured by hand:
  #   FLASH_ATTENTION + KV_CACHE_TYPE  part of what buys full GPU offload
  #   KEEP_ALIVE=-1                    the model used to unload after 5 min
  #                                    idle and the next turn paid a measured
  #                                    14.9s reload
  export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
  export OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}"
  export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"
  # ONE MODEL, ONE SLOT. Both default to more than that, and on a 4GB card both
  # multiply the thing that does not fit: a second resident model holds its own
  # weights, and a second parallel slot holds its own KV cache. Either one puts
  # the card back over the line where Windows WDDM spills to system RAM over
  # PCIe - the silent 114s turn the Modelfile's num_gpu note describes. This
  # server takes one call at a time, so neither buys anything to trade away.
  export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
  export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
  say "Starting Ollama (flash attention, q8_0 KV cache, keep-alive forever, single slot)"
  ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
  for _ in $(seq 1 30); do ollama_up && break; sleep 1; done
  ollama_up || warn "Ollama did not come up - see logs/ollama.log."
fi

if [[ -z "${SKIP_OLLAMA:-}" ]] && ollama_up; then
  # An Ollama that was already running has whatever environment it was started
  # with, and there is no way to read that back over the API - so this stays a
  # warning for that case.
  for var in OLLAMA_FLASH_ATTENTION OLLAMA_KV_CACHE_TYPE OLLAMA_KEEP_ALIVE \
             OLLAMA_MAX_LOADED_MODELS OLLAMA_NUM_PARALLEL; do
    if [[ -z "${!var:-}" ]]; then
      warn "$var is not set for the already-running Ollama - see HANDOFF.md §3."
    fi
  done

  TAGS="$(curl -fsS -m 5 "$OLLAMA_URL/api/tags" || true)"
  REBUILD_FOR_PEAK=""

  # ---- clear the card before measuring it -------------------------------
  #
  # Free VRAM decides whether this run gets full GPU offload (31.3 tok/s) or
  # Ollama's conservative split (12.9 tok/s), and the reading is worthless
  # while something Ollama loaded on a PREVIOUS run is still resident. With
  # OLLAMA_KEEP_ALIVE=-1 - which this script sets, deliberately - that is the
  # normal state, and it is self-defeating: the leftover model's own weights
  # and KV cache are counted as "in use", the check below sees too little free
  # memory, and the run settles for the slow split on a card that was actually
  # free. Worse, the previous copy may be the model we are about to REBUILD,
  # in which case the rebuild lands but the old one keeps answering.
  #
  # Unloading is a keep_alive of 0 against each resident model. It touches
  # nothing but Ollama's own models - never another process's memory - and
  # costs nothing, because the model this call needs is preloaded again a few
  # lines below and every other one was not wanted anyway.
  RESIDENT="$(curl -fsS -m 5 "$OLLAMA_URL/api/ps" 2>/dev/null \
              | tr ',{}' '\n\n\n' | grep -o '"model":"[^"]*"' | cut -d'"' -f4 || true)"
  if [[ -n "$RESIDENT" ]]; then
    say "Unloading resident model(s) to free the GPU: $(tr '\n' ' ' <<<"$RESIDENT")"
    while read -r loaded; do
      [[ -z "$loaded" ]] && continue
      curl -fsS -m 10 "$OLLAMA_URL/api/generate" \
        -d "{\"model\":\"$loaded\",\"keep_alive\":0}" >/dev/null 2>&1 || true
    done <<<"$RESIDENT"
    # The unload is asynchronous; nvidia-smi lags it by a moment.
    for _ in $(seq 1 10); do
      curl -fsS -m 5 "$OLLAMA_URL/api/ps" 2>/dev/null | grep -q '"model"' || break
      sleep 1
    done
  fi
  # On an unloaded model, free VRAM is the real capacity available to it. If
  # this card has the measured 3400 MiB headroom and the operator did not set a
  # different layer count, bake in full offload: it measured 31.3 tok/s versus
  # 12.9 tok/s with Ollama's conservative split. Below the threshold, leave it
  # adaptive; forcing 99 there makes every request fail instead of degrading.
  # The "already loaded" guard the unload above replaces: with nothing
  # resident, the reading is the card's real capacity every run, so a machine
  # that was too busy last time can pick up full offload on this one.
  if [[ -z "${LLM_NUM_GPU:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
                | head -1 | tr -dc '0-9')"
    if [[ -n "${FREE_MIB:-}" ]] && (( FREE_MIB >= 3400 )); then
      export LLM_NUM_GPU=99
      if grep -q "\"$MODEL_NAME" <<<"$TAGS" \
         && ! ollama show "$MODEL_NAME" --modelfile 2>/dev/null \
              | grep -Eq '^PARAMETER[[:space:]]+num_gpu[[:space:]]+99([[:space:]]|$)'; then
        REBUILD_FOR_PEAK=1
      fi
      say "${FREE_MIB} MiB VRAM free; selecting full LLM GPU offload"
    else
      say "${FREE_MIB:-unknown} MiB VRAM free; keeping Ollama's adaptive GPU fit"
      # NAME WHAT IS HOLDING THE CARD. "Close optional GPU applications" is
      # advice nobody can act on - the culprits are WebView2 helpers with names
      # like msedgewebview2.exe, four of them, none obviously belonging to the
      # chat app that spawned it. Measured, ~1.2 GB of this card was held that
      # way, which is the whole difference between 12.9 and 31.3 tok/s.
      #
      # Only the rows that carry a real number. On Windows the driver is in
      # WDDM mode and nvidia-smi cannot see per-process VRAM at all - every row
      # comes back "[N/A], C:\...\SearchHost.exe", which is worse than silence
      # because it names innocent system processes as the culprit. Task Manager
      # DOES have the figure there, under Details > Dedicated GPU memory, so
      # that is where this points instead of printing a list it cannot rank.
      HOGS="$(nvidia-smi --query-compute-apps=used_memory,process_name \
                --format=csv,noheader,nounits 2>/dev/null \
              | grep -E '^[0-9]+,' | sort -rn | head -5)"
      if [[ -n "${HOGS:-}" ]]; then
        warn "Holding VRAM right now (close these for the 2.2x, then rerun):"
        while read -r hog; do [[ -n "$hog" ]] && warn "  ${hog} MiB"; done <<<"$HOGS"
      else
        warn "  Need 3400 MiB free for full offload. Task Manager > Details >"
        warn "  Dedicated GPU memory shows who has it (usually msedgewebview2.exe)."
      fi
    fi
  fi

  if grep -q "\"$MODEL_NAME" <<<"$TAGS" && [[ -z "$REBUILD_FOR_PEAK" ]]; then
    say "Ollama has $MODEL_NAME"
  elif command -v ollama >/dev/null 2>&1; then
    # Pulled explicitly rather than left to `ollama create`, so a fresh machine
    # gets download progress on a multi-GB base model instead of a silent wait.
    if [[ -n "$BASE_MODEL" ]] && ! grep -q "\"$BASE_MODEL\"" <<<"$TAGS"; then
      say "Pulling the base model $BASE_MODEL (one time, a few GB)"
      ollama pull "$BASE_MODEL"
    fi
    # Built through setup_model.py rather than `ollama create -f Modelfile`,
    # so there is ONE build path. There used to be two, and they disagreed:
    # this line pinned num_ctx from the Modelfile while setup_model.py pinned
    # its own 8192, and whichever had been run last decided what Ollama really
    # served. Both now read LLM_NUM_CTX / LLM_NUM_GPU out of .env.
    if [[ -n "$REBUILD_FOR_PEAK" ]]; then
      say "Rebuilding $MODEL_NAME for full GPU offload"
    else
      say "Building $MODEL_NAME (one time)"
    fi
    "$PY" -m backend.scripts.setup_model --target "$MODEL_NAME"
  else
    warn "$MODEL_NAME is not in Ollama and the 'ollama' CLI is not on PATH."
    warn "  ollama create $MODEL_NAME -f Modelfile"
  fi

  # Load the model NOW, before the first caller. Ollama otherwise waits until
  # the first chat request to allocate weights/KV cache; measured cold-start
  # cost on this machine was 14.9 seconds. An empty generate request is the
  # documented load-only path, and keep_alive=-1 prevents idle eviction.
  say "Preloading $MODEL_NAME"
  if ! "$PY" - "$OLLAMA_URL" "$MODEL_NAME" <<'PY'
import json
import sys
import urllib.request

url, model = sys.argv[1:]
request = urllib.request.Request(
    f"{url.rstrip('/')}/api/generate",
    data=json.dumps({"model": model, "prompt": "", "stream": False, "keep_alive": -1}).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=300) as response:
    if response.status != 200:
        raise SystemExit(f"Ollama preload returned HTTP {response.status}")
PY
  then
    die "Could not preload $MODEL_NAME. The first caller would hit a broken or cold LLM; see Ollama's log."
  fi

  # This is the measured peak path for the 4 GB RTX 2050. Report it rather
  # than silently accepting a CPU/GPU split: a split works, but generation was
  # 12.9 tok/s versus 31.3 tok/s at 100% GPU. Do not kill unrelated desktop
  # applications here; the operator decides what may be closed.
  if command -v ollama >/dev/null 2>&1; then
    MODEL_RESIDENCY="$(ollama ps 2>/dev/null | grep -F "$MODEL_NAME" || true)"
    if grep -q "100% GPU" <<<"$MODEL_RESIDENCY"; then
      say "$MODEL_NAME is resident at 100% GPU"
    elif [[ -n "$MODEL_RESIDENCY" ]]; then
      warn "$MODEL_NAME is loaded but not at 100% GPU; generation will be slower."
      warn "  Close optional GPU applications, set LLM_NUM_GPU=99, rebuild, and rerun."
    else
      die "$MODEL_NAME did not remain resident after preload."
    fi
  fi

  # THE VRAM PRECONDITION, which is not self-enforcing. Full offload measured
  # 2.2x (31.3 vs 12.9 tok/s), and that is only true while the card is free:
  # with ~1.2 GB held by desktop apps the weights fit and the weights PLUS the
  # KV cache did not, and Windows WDDM does not fail that allocation - it
  # spills to system RAM over PCIe, silently, and one turn measured 114s.
  #
  # Only worth warning about when LLM_NUM_GPU is set. Left blank (the default)
  # llama.cpp fits the model to whatever is free and the failure mode is a
  # slower split, not a stall - see LLM_NUM_GPU in .env.example.
  if [[ -n "${LLM_NUM_GPU:-}" ]] && command -v nvidia-smi >/dev/null 2>&1 \
     && ! ollama ps 2>/dev/null | grep -q "$MODEL_NAME"; then
    FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
                | head -1 | tr -dc '0-9')"
    if [[ -n "${FREE_MIB:-}" ]] && (( FREE_MIB < 3400 )); then
      warn "Only ${FREE_MIB} MiB of VRAM free and LLM_NUM_GPU=$LLM_NUM_GPU is forcing full offload."
      warn "  Close other GPU apps, or blank LLM_NUM_GPU in .env and rebuild the model."
      warn "  Forced offload that does not fit is a 500 on every turn, not a slow turn."
    fi
  fi
elif [[ -z "${SKIP_OLLAMA:-}" ]]; then
  warn "Ollama is not answering at $OLLAMA_URL - the agent cannot reply."
  warn "  Install it from https://ollama.com/download, then run this again."
fi

# ----------------------------------------------------------------- serve ----
# A SERVER ALREADY ON THIS PORT IS A TRAP, not an inconvenience. It caches the
# prompts and the code it started with, so a stale one answers every request
# and every check passes against code that is no longer on disk. That has cost
# a debugging session; uvicorn's own bind error scrolls past in a log file.
if curl -fsS -m 3 "http://127.0.0.1:$BACKEND_PORT/api/health" >/dev/null 2>&1; then
  die "Something is already serving on port $BACKEND_PORT, and it is running the
     code it STARTED with, not the code on disk. Stop it first:
       Windows:  Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" |
                   Where-Object { \$_.CommandLine -like '*uvicorn*' } |
                   ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }
       Linux:    pkill -f 'uvicorn backend.main:app'
     Or serve elsewhere:  BACKEND_PORT=9000 ./run.sh"
fi

if [[ "$BACKEND_HOST" != "127.0.0.1" && "$BACKEND_HOST" != "localhost" ]]; then
  # Measured: binding 0.0.0.0 does not get you a working console from another
  # machine. Browsers only allow getUserMedia on localhost or HTTPS, so the
  # microphone dies on a plain-HTTP LAN origin - port-forward instead:
  #   ssh -L $BACKEND_PORT:127.0.0.1:$BACKEND_PORT user@host
  warn "Binding $BACKEND_HOST: the console's microphone will NOT work over plain"
  warn "  HTTP from another machine (getUserMedia needs localhost or HTTPS)."
  warn "  Prefer: ssh -L $BACKEND_PORT:127.0.0.1:$BACKEND_PORT user@host"
  [[ -z "${AUDIO_WS_AUTH_TOKEN:-}" ]] && \
    warn "  AUDIO_WS_AUTH_TOKEN is unset, so /ws/audio has NO auth on that interface."
fi

# stdout AND stderr to a file: several real bugs in this project were only ever
# visible in the server log, and a backgrounded process has no console.
UVICORN_MODE=(--workers 1)
if [[ -n "${RELOAD:-}" ]]; then
  # Reload is explicitly development-only. It adds a supervisor process and
  # drops active calls on code changes, so the peak/default path never uses it.
  UVICORN_MODE=(--reload)
  warn "RELOAD is enabled: development convenience, not peak performance."
fi

say "Starting one API worker on $BACKEND_HOST:$BACKEND_PORT (log: ${LOG_FILE#$ROOT_DIR/})"
"$PY" -m uvicorn backend.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" "${UVICORN_MODE[@]}" \
  > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

cleanup() {
  printf '\n'
  say "Stopping AICA"
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Model loading (ASR especially) takes a while on a cold start; poll rather
# than guess, and fail loudly if the process dies during it.
say "Loading models..."
for _ in $(seq 1 60); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    warn "The server exited during startup. Last lines of the log:"
    tail -20 "$LOG_FILE" >&2
    exit 1
  fi
  HEALTH="$(curl -fsS -m 3 "http://127.0.0.1:$BACKEND_PORT/api/health" 2>/dev/null || true)"
  [[ -n "$HEALTH" ]] && break
  sleep 2
done

if [[ -z "${HEALTH:-}" ]]; then
  warn "The server did not become healthy in time. Last lines of the log:"
  tail -20 "$LOG_FILE" >&2
  exit 1
fi

# Health becomes available as soon as the adapters load, while the fixed TTS
# lines continue warming in the background. Wait for that cache too: otherwise
# run.sh can print READY and the first caller still pays twenty network
# synthesis round-trips. This is an optimisation, so a failed warm is reported
# without taking down an otherwise healthy server.
say "Warming fixed TTS phrases..."
TTS_WARMED=""
for _ in $(seq 1 45); do
  TTS_WARMED="$(grep -E 'TTS cache warmed with [0-9]+ of [0-9]+ fixed lines' "$LOG_FILE" | tail -1 || true)"
  [[ -n "$TTS_WARMED" ]] && break
  sleep 1
done
if [[ -n "$TTS_WARMED" ]]; then
  say "$TTS_WARMED"
else
  warn "TTS cache warm did not finish within 45 seconds; live synthesis is still available."
fi

# A server that answers requests is not evidence that ASR, the LLM or TTS
# loaded - say which of the four actually did.
printf '\n'
"$PY" - "$HEALTH" <<'EOF'
import json, sys
h = json.loads(sys.argv[1])
labels = [("asr_ready", "speech-to-text"), ("conversation_ready", "conversation"),
          ("llm_ready", "LLM"), ("tts_ready", "voice")]
for key, label in labels:
    ok = bool(h.get(key))
    print(f"    {'OK  ' if ok else 'DOWN'}  {label}" + ("" if ok else "   (see logs/server.log)"))
print(f"\n    model: {h.get('llm_model')} at {h.get('llm_base_url')}")
EOF

cat <<EOF

    Console:  http://localhost:$BACKEND_PORT/console

    Ctrl+C to stop.

EOF

wait "$SERVER_PID"
