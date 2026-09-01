"""Build the `aruvi-base` Ollama model the backend's .env points at.

Why this exists: LLM_MODEL must name a model whose num_ctx is set EXPLICITLY.
Pointing it at a bare tag lets Ollama pick a VRAM-derived default (often
2048-4096), which silently truncates the assembled system prompt before its
language rules and produces the two failures captured in LLM_TEST_RESULTS.txt -
the agent answers in English and invents a caller mobile number. The repo-root
Modelfile pins num_ctx 8192, but it hard-codes a `FROM` tag that may not be
pulled on a given machine; when it isn't, the backend starts fine and then
fails on the first turn with a 404 from Ollama, which looks nothing like a
configuration problem.

So this picks the best base actually present, creates `aruvi-base` from it, and
says which one it used. Preference order is quality-first: the 4B instruct
model is the one the transcripts in LLM_TEST_RESULTS.txt were captured on and
is the smallest that held the Tamil/English register; the fallbacks below it
are progressively weaker but keep a fresh machine testable rather than broken.

Usage:
    python -m backend.scripts.setup_model            # create/refresh aruvi-base
    python -m backend.scripts.setup_model --list     # just show what is available
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from backend.settings import LlmSettings
import urllib.error
import urllib.request

# Ollama's REST port. Kept separate from LLM_BASE_URL (which points at the
# OpenAI-compatible /v1 surface) because model management is not part of that
# API - creating a model is an Ollama-native operation.
OLLAMA_HOST = "http://127.0.0.1:11434"

TARGET_MODEL = "aruvi-base"

# READ FROM .env, not restated here. This script is what actually BUILDS the
# model, and it used to carry its own NUM_CTX = 8192 while both the Modelfile
# and LLM_NUM_CTX said 6144. The two that were checked against each other
# agreed; the one that did the building did not - so the model Ollama actually
# served had a window nothing in the repo claimed, visible only in `ollama ps`.
# Deriving these removes the drift instead of adding a third guard against it.
_LLM = LlmSettings()
NUM_CTX = _LLM.num_ctx
TEMPERATURE = _LLM.temperature
NUM_GPU = _LLM.num_gpu

# Best first. Every entry must support OpenAI-style tool calling, because the
# whole tool layer (backend/tools.py) depends on it - a model without the
# "tools" capability will hold a conversation and never book anything.
PREFERRED_BASES = [
    "qwen3:4b-instruct-2507-q4_K_M",
    "qwen2.5:3b",
    "mistral:latest",
    "llama3.2:latest",
]


def installed_models() -> list[dict]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=10) as response:
            return json.load(response).get("models", [])
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"Could not reach Ollama at {OLLAMA_HOST} ({error}).\n"
            "Start it with `ollama serve` (or launch the Ollama app) and try again."
        )


def pick_base(models: list[dict]) -> str | None:
    names = {model["name"] for model in models}
    # Ollama reports "qwen2.5:3b" for a model pulled as "qwen2.5:3b" but some
    # tags round-trip with an explicit ":latest"; accept either spelling.
    for candidate in PREFERRED_BASES:
        if candidate in names:
            return candidate
        if ":" not in candidate and f"{candidate}:latest" in names:
            return f"{candidate}:latest"
    return None


def supports_tools(models: list[dict], name: str) -> bool:
    for model in models:
        if model["name"] == name:
            return "tools" in (model.get("capabilities") or [])
    return False


def build_modelfile(base: str) -> str:
    """The Modelfile this build uses. .env is the only source of truth for it."""
    lines = [
        f"FROM {base}",
        f"PARAMETER num_ctx {NUM_CTX}",
        f"PARAMETER temperature {TEMPERATURE}",
    ]
    # Omitted ENTIRELY when blank, rather than written as some default. An
    # explicit num_gpu disables llama.cpp's own fit-to-free-VRAM logic
    # ("n_gpu_layers already set by user to 99, abort"), which turns a machine
    # that is merely short on VRAM into one that 500s on every single turn.
    # See LlmSettings.num_gpu for the measurement behind that.
    if NUM_GPU:
        lines.append(f"PARAMETER num_gpu {NUM_GPU}")
    return "\n".join(lines) + "\n"


def create_model(base: str) -> None:
    modelfile = build_modelfile(base)
    offload = f"num_gpu={NUM_GPU}" if NUM_GPU else "GPU offload chosen by Ollama"
    print(f"Creating {TARGET_MODEL} from {base} (num_ctx={NUM_CTX}, {offload})...")

    # A real file, not `-f -`: passing the Modelfile on stdin is rejected by
    # released Ollama builds ("no Modelfile or safetensors files found").
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "Modelfile"
        path.write_text(modelfile, encoding="utf-8")
        result = subprocess.run(
            ["ollama", "create", TARGET_MODEL, "-f", str(path)],
            text=True,
            capture_output=True,
        )
    if result.returncode != 0:
        raise SystemExit(
            f"`ollama create` failed:\n{result.stderr.strip()}\n\n"
            "If `ollama` is not on PATH, run it manually with the repo-root Modelfile:\n"
            f"    ollama create {TARGET_MODEL} -f Modelfile"
        )
    print(result.stdout.strip() or "done")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="show installed models and exit")
    args = parser.parse_args()

    models = installed_models()

    if args.list:
        print(f"Models installed at {OLLAMA_HOST}:")
        for model in models:
            caps = ",".join(model.get("capabilities") or []) or "-"
            print(f"  {model['name']:<40} capabilities={caps}")
        return

    base = pick_base(models)
    if base is None:
        raise SystemExit(
            "None of the supported base models are installed. Pull the preferred one:\n"
            f"    ollama pull {PREFERRED_BASES[0]}\n"
            f"then re-run this script. (Also acceptable: {', '.join(PREFERRED_BASES[1:])})"
        )

    if not supports_tools(models, base):
        print(
            f"WARNING: {base} does not advertise tool-calling support. The conversation "
            "will run but no tool (lookupPatient, bookAppointment, ...) will ever fire.",
            file=sys.stderr,
        )

    if base != PREFERRED_BASES[0]:
        print(
            f"NOTE: falling back to {base}; {PREFERRED_BASES[0]} is the model the "
            "transcripts in LLM_TEST_RESULTS.txt were captured on and holds the "
            "Tamil/English register better. Pull it when you can:\n"
            f"    ollama pull {PREFERRED_BASES[0]}\n"
        )

    create_model(base)
    print(
        f"\nDone. Set this in .env (it is the default):\n"
        f"    LLM_MODEL={TARGET_MODEL}\n"
        f"    LLM_BASE_URL={OLLAMA_HOST}/v1\n"
        "Then start the backend:  uvicorn backend.main:app --reload"
    )


if __name__ == "__main__":
    main()
