""".env.example must document every knob the code actually reads.

The promise this file guards is "you can change anything from .env". That
promise is not kept by the code being configurable - it is kept by the
configurable things being FINDABLE. A knob added to settings.py and never
written down is, in practice, not configurable: nobody knows it exists.

This is the cheap half of that. It cannot check that the documentation is
GOOD, only that it exists, which is the part that rots silently.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_EXAMPLE = _REPO_ROOT / ".env.example"
_BACKEND = _REPO_ROOT / "backend"

# `os.getenv("NAME"` / `os.environ["NAME"` anywhere in the package.
_GETENV_RE = re.compile(r"""os\.(?:getenv\(|environ\[)["']([A-Z][A-Z0-9_]*)["']""")

# `${NAME:-default}` and `${NAME:+...}` in run.sh - the shell half of the same
# config surface. Reading a name with a default is exactly the shape of
# "overridable from outside", which is what a knob is.
_SHELL_RE = re.compile(r"\$\{([A-Z][A-Z0-9_]*):[-+]")

# ...but run.sh also reads its OWN locals that way, purely as a `set -u` guard
# (`${HEALTH:-}` after a curl that may have produced nothing). Those are
# assigned from something that is not themselves, which is what separates them
# from a real knob - a knob is only ever assigned as NAME="${NAME:-default}".
# Detected rather than listed, so a local added later does not fail this test.
_SHELL_ASSIGN_RE = re.compile(r"^\s*([A-Z][A-Z0-9_]*)=(.*)$", re.MULTILINE)


def _shell_locals(text: str) -> set[str]:
    return {
        name
        for name, value in _SHELL_ASSIGN_RE.findall(text)
        if not value.lstrip('"').startswith("${" + name + ":")
    }


def _documented_names() -> set[str]:
    """Every NAME= in .env.example, whether the line is commented out or not."""
    text = _ENV_EXAMPLE.read_text(encoding="utf-8")
    return set(re.findall(r"^\s*#?\s*([A-Z][A-Z0-9_]*)=", text, re.MULTILINE))


def _python_names() -> dict[str, set[str]]:
    """env var -> the backend/*.py files that read it. Tests are not code."""
    found: dict[str, set[str]] = {}
    for path in sorted(_BACKEND.rglob("*.py")):
        if path.name.startswith("test_") or "__pycache__" in path.parts:
            continue
        for name in _GETENV_RE.findall(path.read_text(encoding="utf-8")):
            found.setdefault(name, set()).add(path.relative_to(_REPO_ROOT).as_posix())
    return found


def test_env_example_documents_every_variable_the_backend_reads() -> None:
    documented = _documented_names()
    undocumented = {
        name: sorted(where)
        for name, where in _python_names().items()
        if name not in documented
    }
    assert not undocumented, (
        "These env vars are read by the code but are not in .env.example, so "
        "nobody deploying this can discover them:\n"
        + "\n".join(f"  {name}  ({', '.join(where)})" for name, where in sorted(undocumented.items()))
        + "\n\nAdd each one to .env.example, commented at its default, with a "
        "line saying what changing it costs."
    )


def test_env_example_documents_every_variable_run_sh_reads() -> None:
    """run.sh reads .env too, so its knobs belong in the same file."""
    run_sh = _REPO_ROOT / "run.sh"
    if not run_sh.exists():  # pragma: no cover - run.sh is committed
        pytest.skip("run.sh not present")

    text = run_sh.read_text(encoding="utf-8")
    documented = _documented_names()
    internal = _shell_locals(text)
    undocumented = sorted(
        name
        for name in _SHELL_RE.findall(text)
        if name not in documented and name not in internal
    )
    assert not undocumented, (
        "run.sh reads these with a default but .env.example does not mention "
        f"them: {undocumented}"
    )


def test_env_example_declares_no_variable_that_nothing_reads() -> None:
    """The other direction: a documented knob that does nothing is a lie.

    A renamed setting leaves its old name behind in .env.example, where it
    reads as a working knob for as long as nobody checks.
    """
    read_somewhere = set(_python_names())
    run_sh = (_REPO_ROOT / "run.sh").read_text(encoding="utf-8")
    read_somewhere |= set(_SHELL_RE.findall(run_sh)) - _shell_locals(run_sh)
    # run.sh exports these into Ollama's environment rather than reading them
    # for itself, so they never appear as ${NAME:-...} on the read side.
    read_somewhere |= {"OLLAMA_FLASH_ATTENTION", "OLLAMA_KV_CACHE_TYPE", "OLLAMA_KEEP_ALIVE"}

    dead = sorted(_documented_names() - read_somewhere)
    assert not dead, (
        f".env.example documents these, but nothing reads them: {dead}. "
        "Either wire them up or delete the lines - a knob that does nothing "
        "is worse than a missing one, because it looks like it worked."
    )


def test_the_shipped_env_file_is_parseable_and_has_no_unknown_keys() -> None:
    """A real .env, if present, must only set things the code understands.

    A typo'd key in .env is completely silent today: dotenv loads it, nothing
    reads it, and the setting you thought you changed keeps its default.
    """
    env_file = _REPO_ROOT / ".env"
    if not env_file.exists():
        pytest.skip("no .env on this machine")

    documented = _documented_names()
    unknown = []
    for number, line in enumerate(env_file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=", line)
        assert match, f".env line {number} is not KEY=VALUE: {line!r}"
        if match.group(1) not in documented:
            unknown.append(f"line {number}: {match.group(1)}")

    assert not unknown, (
        "These keys are set in .env but no such setting exists - almost always "
        f"a typo, and a silent one: {unknown}"
    )
