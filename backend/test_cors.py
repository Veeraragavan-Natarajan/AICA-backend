"""The dashboard is served from a DIFFERENT origin than this backend.

Without CORS its fetch of /api/health and /api/calls is blocked by the browser
and the whole dashboard reads as "backend down" - a failure with no server-side
symptom at all, which is why it is worth a test rather than a manual check.

Run in a SUBPROCESS, not with monkeypatch. main.py adds the middleware at
import time, because Starlette builds its middleware stack on the first request
and anything appended after that is silently ignored. So the only way to
exercise a given CORS_ALLOW_ORIGINS is to import main.py in a process that had
it set from the start - which is also exactly how a deployment works.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Imports main.py, fires one cross-origin GET at /api/health, and prints the
# response headers. TestClient without the `with` form never runs lifespan(),
# so no model loads - but /api/health reads app.state, so it is stubbed.
_PROBE = """
import json, sys
from fastapi.testclient import TestClient
from backend.main import app

class _Ready:
    ready = True
    settings = type("S", (), {"model": "m", "base_url": "u"})()

app.state.asr = _Ready()
app.state.conversation = _Ready()
app.state.llm = _Ready()
app.state.tts = _Ready()

client = TestClient(app)
response = client.get("/api/health", headers={"Origin": sys.argv[1]})
print(json.dumps({
    "status": response.status_code,
    "allow_origin": response.headers.get("access-control-allow-origin"),
    "allow_credentials": response.headers.get("access-control-allow-credentials"),
}))
"""


def _probe(origin: str, cors_setting: str) -> dict:
    # Set EXPLICITLY, empty string included: dotenv does not override a name
    # already in the environment, so this pins the case under test whatever
    # this machine's .env happens to say.
    env = dict(os.environ, CORS_ALLOW_ORIGINS=cors_setting)
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, origin],
        cwd=_REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(f"probe failed:\n{result.stdout}\n{result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_a_configured_origin_is_allowed_to_read_the_api() -> None:
    result = _probe("http://localhost:5173", "http://localhost:5173")
    assert result["status"] == 200
    assert result["allow_origin"] == "http://localhost:5173", (
        "The dashboard's own origin was refused, which the browser reports as a "
        "dead backend and the server log shows as a perfectly normal 200."
    )


def test_an_origin_that_was_never_configured_gets_no_grant() -> None:
    result = _probe("https://somewhere-else.example", "http://localhost:5173")
    assert result["allow_origin"] is None, (
        "An unlisted origin was granted access to the call log, which is "
        "patient data."
    )


def test_a_comma_separated_list_admits_every_entry() -> None:
    both = "http://localhost:5173, https://aruvi.example.com"
    for origin in ("http://localhost:5173", "https://aruvi.example.com"):
        assert _probe(origin, both)["allow_origin"] == origin, (
            f"{origin} was dropped - whitespace around a comma must not matter, "
            "because a human edits this list by hand in .env."
        )


def test_cors_is_off_entirely_when_nothing_is_configured() -> None:
    """The default. A console-only deployment should grant no origin at all."""
    result = _probe("http://localhost:5173", "")
    assert result["status"] == 200, "same-origin /console traffic must be unaffected"
    assert result["allow_origin"] is None


def test_a_wildcard_never_ships_with_credentials() -> None:
    """`*` plus credentials is rejected by every browser, silently.

    Setting CORS_ALLOW_ORIGINS=* on a closed network is a legitimate choice;
    pairing it with allow_credentials would make every request fail with no
    server-side symptom, so the two must not be on together.
    """
    result = _probe("https://anything.example", "*")
    assert result["allow_origin"] == "*"
    assert result["allow_credentials"] != "true"
