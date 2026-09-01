"""Regression coverage for the production Ollama model-build path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from .scripts import setup_model


def test_create_model_uses_the_requested_target(monkeypatch) -> None:
    """run.sh's LLM_MODEL and the name passed to `ollama create` must agree."""
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="done", stderr="")

    monkeypatch.setattr(setup_model.subprocess, "run", fake_run)

    setup_model.create_model("some-base:tag", "deployment-model")

    assert len(calls) == 1
    assert calls[0][:3] == ["ollama", "create", "deployment-model"]


def test_run_sh_passes_its_configured_model_name_to_the_builder() -> None:
    run_sh = (Path(__file__).resolve().parent.parent / "run.sh").read_text(encoding="utf-8")
    assert 'backend.scripts.setup_model --target "$MODEL_NAME"' in run_sh
