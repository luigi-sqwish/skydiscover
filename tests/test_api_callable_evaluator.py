"""Regression tests for the public Python API."""

import pytest

import skydiscover.api as api_module
from skydiscover.api import DiscoveryResult, run_discovery
from skydiscover.config import Config, LLMModelConfig


def test_run_discovery_callable_evaluator_receives_split_kwargs():
    calls = []

    def evaluator(program_path, *, split="train", phase="search"):
        calls.append((split, phase))
        score = 0.1 if split == "train" else 0.9
        if phase == "final":
            score += 0.05
        return {"combined_score": score}

    config = Config(max_iterations=0)
    config.llm.models = [LLMModelConfig(name="gpt-5", api_key="test-key")]
    config.llm.evaluator_models = config.llm.models.copy()
    config.llm.guide_models = config.llm.models.copy()

    result = run_discovery(
        evaluator=evaluator,
        initial_program="def solve():\n    return 1",
        config=config,
        iterations=0,
        task_mode="generalization",
        val_split="val",
    )

    assert result.best_program is not None
    assert result.best_score == pytest.approx(0.95)
    assert result.best_program.metrics["combined_score"] == pytest.approx(0.9)
    assert result.best_program.metrics["final_combined_score"] == pytest.approx(0.95)
    assert calls == [("train", "search"), ("val", "search"), ("val", "final")]


def test_run_discovery_preserves_legacy_positional_config_binding(monkeypatch):
    captured = {}

    async def fake_run_discovery_async(initial_program, evaluator, config, **kwargs):
        captured["initial_program"] = initial_program
        captured["evaluator"] = evaluator
        captured["config"] = config
        captured.update(kwargs)
        return DiscoveryResult(
            best_program=None,
            best_score=0.0,
            best_solution="",
            metrics={},
            output_dir=None,
        )

    monkeypatch.setattr(api_module, "_run_discovery_async", fake_run_discovery_async)

    config = Config()
    result = run_discovery(
        "eval.py",
        "init.py",
        "gpt-5",
        10,
        "topk",
        config,
        True,
        "/tmp/out",
        "system prompt",
        "https://example.test/v1",
        False,
        task_mode="generalization",
        val_split="val",
    )

    assert result.best_score == 0.0
    assert captured["initial_program"] == "init.py"
    assert captured["evaluator"] == "eval.py"
    assert captured["config"] is config
    assert captured["agentic"] is True
    assert captured["output_dir"] == "/tmp/out"
    assert captured["system_prompt"] == "system prompt"
    assert captured["api_base"] == "https://example.test/v1"
    assert captured["cleanup"] is False
    assert captured["task_mode"] == "generalization"
    assert captured["val_split"] == "val"
