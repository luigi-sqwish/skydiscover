"""Tests for the shared ALMA benchmark bridge."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from skydiscover.config import EvaluatorConfig
from skydiscover.evaluation import create_evaluator
from skydiscover.evaluation.alma_bridge import (
    ALMA_ALFWORLD_TASK,
    ALMA_BABAISAI_TASK,
    ALMA_MINIHACK_TASK,
    ALMA_TEXTWORLD_TASK,
    make_alma_benchmark_adapter,
    resolve_alma_root,
    resolve_alma_split,
    summarize_feedback,
)


def _write_fake_alma_root(
    base_dir: Path,
    *,
    score_map: dict[str, float] | None = None,
    record_path: Path | None = None,
    broken_import: bool = False,
) -> Path:
    alma_root = base_dir / "alma"
    (alma_root / "core").mkdir(parents=True)
    (alma_root / "core" / "__init__.py").write_text("", encoding="utf-8")
    (alma_root / "eval_in_container.py").write_text(
        "async def run_evaluation(*args, **kwargs):\n    return None\n",
        encoding="utf-8",
    )

    if broken_import:
        (alma_root / "core" / "memo_manager.py").write_text(
            "raise RuntimeError('broken import')\n",
            encoding="utf-8",
        )
        return alma_root

    score_map_json = json.dumps(
        score_map
        or {
            "train": 0.73,
            "eval_in_distribution": 0.81,
            "eval_out_of_distribution": 0.67,
        }
    )
    record_literal = repr(str(record_path)) if record_path else "None"
    (alma_root / "core" / "memo_manager.py").write_text(
        textwrap.dedent(
            f"""
            import json
            from pathlib import Path

            SCORE_MAP = json.loads({score_map_json!r})
            RECORD_PATH = {record_literal}

            class Memo_Manager:
                def __init__(self, task_type, **kwargs):
                    self.task_type = task_type

                async def execute_memo_structure(self, **kwargs):
                    if RECORD_PATH:
                        path = Path(RECORD_PATH)
                        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
                        existing.append({{
                            "task_type": self.task_type,
                            "mode": kwargs.get("mode"),
                            "eval_type": kwargs.get("eval_type"),
                            "status": kwargs.get("status"),
                            "update_task": kwargs.get("update_task"),
                            "train_size": kwargs.get("train_size"),
                            "update_size": kwargs.get("update_size"),
                        }})
                        path.write_text(json.dumps(existing), encoding="utf-8")

                    score = float(SCORE_MAP.get(kwargs.get("status"), 0.0))
                    data = {{
                        "benchmark_eval_score": {{
                            "benchmark_overall_eval_score": score,
                            "benchmark_overall_eval_standard_deviation": 0.05,
                        }},
                        "examples": [
                            {{
                                "init_environment": {{
                                    "goal": "put the apple in the fridge",
                                    "obs": "Kitchen with a closed fridge.",
                                }},
                                "memo_retrieved": {{}},
                                "final_reward": 0.0,
                            }},
                            {{
                                "error_info": {{
                                    "error_type": "RuntimeError",
                                    "message": "boom",
                                }},
                                "final_reward": 0.0,
                            }},
                        ],
                        "token_usage": {{
                            "gpt-4o-mini": {{
                                "total_tokens": 20,
                                "prompt_tokens": 12,
                                "completion_tokens": 8,
                                "completion_tokens_details": {{
                                    "reasoning_tokens": 3,
                                }},
                            }},
                        }},
                    }}
                    success = kwargs.get("mode") == "test"
                    return success, data, "fake-sha", kwargs.get("code_str", "")
            """
        ),
        encoding="utf-8",
    )
    return alma_root


async def _async_result(value):
    return value


def _candidate_file(tmp_path: Path) -> Path:
    candidate_path = tmp_path / "candidate.py"
    candidate_path.write_text("class Candidate: pass\n", encoding="utf-8")
    return candidate_path


def _bridge_eval_file(tmp_path: Path) -> Path:
    evaluator_path = tmp_path / "bridge_eval.py"
    evaluator_path.write_text(
        textwrap.dedent(
            """
            from skydiscover.evaluation.alma_bridge import (
                ALMA_TEXTWORLD_TASK,
                make_alma_benchmark_adapter,
            )

            _ADAPTER = make_alma_benchmark_adapter(ALMA_TEXTWORLD_TASK, benchmark_file=__file__)

            def resolve_agentic_codebase_root():
                return _ADAPTER.resolve_agentic_codebase_root()

            def validate_config(config):
                _ADAPTER.validate_config(config)

            def evaluate_stage1(program_path, *, split="train", phase="search"):
                return _ADAPTER.evaluate_stage1(program_path, split=split, phase=phase)

            def evaluate_stage2(program_path, *, split="train", phase="search"):
                return _ADAPTER.evaluate_stage2(program_path, split=split, phase=phase)

            def evaluate(program_path, *, split="train", phase="search"):
                return _ADAPTER.evaluate(program_path, split=split, phase=phase)
            """
        ),
        encoding="utf-8",
    )
    return evaluator_path


def test_resolve_alma_root_prefers_env_override(tmp_path):
    alma_root = _write_fake_alma_root(tmp_path)

    resolved = resolve_alma_root({"ALMA_ROOT": str(alma_root)})

    assert resolved == alma_root.resolve()


def test_resolve_alma_root_supports_sibling_fallback(tmp_path):
    benchmark_file = tmp_path / "skydiscover" / "benchmarks" / "alma_textworld" / "evaluator.py"
    benchmark_file.parent.mkdir(parents=True)
    benchmark_file.write_text("", encoding="utf-8")
    alma_root = _write_fake_alma_root(tmp_path)

    resolved = resolve_alma_root({}, benchmark_file=benchmark_file)

    assert resolved == alma_root.resolve()


def test_resolve_alma_root_reports_invalid_layout(tmp_path):
    alma_root = tmp_path / "alma"
    alma_root.mkdir()

    with pytest.raises(FileNotFoundError, match="ALMA memo manager"):
        resolve_alma_root({"ALMA_ROOT": str(alma_root)})


def test_validate_config_rejects_missing_docker(monkeypatch, tmp_path):
    adapter = make_alma_benchmark_adapter(ALMA_TEXTWORLD_TASK)
    alma_root = _write_fake_alma_root(tmp_path)
    monkeypatch.setenv("ALMA_ROOT", str(alma_root))
    monkeypatch.setattr("shutil.which", lambda name: None)

    with pytest.raises(RuntimeError, match="Docker executable not found"):
        adapter.validate_config(EvaluatorConfig(task_mode="multi_task"))


def test_validate_config_surfaces_importability_failures(monkeypatch, tmp_path):
    adapter = make_alma_benchmark_adapter(ALMA_TEXTWORLD_TASK)
    alma_root = _write_fake_alma_root(tmp_path, broken_import=True)
    monkeypatch.setenv("ALMA_ROOT", str(alma_root))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")

    with pytest.raises(ImportError, match="Failed to import ALMA Memo_Manager"):
        adapter.validate_config(EvaluatorConfig(task_mode="single_task"))


@pytest.mark.parametrize("descriptor", (ALMA_TEXTWORLD_TASK, ALMA_BABAISAI_TASK))
def test_validate_config_rejects_empty_ood_splits(monkeypatch, tmp_path, descriptor):
    adapter = make_alma_benchmark_adapter(descriptor)
    alma_root = _write_fake_alma_root(tmp_path)
    monkeypatch.setenv("ALMA_ROOT", str(alma_root))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")

    with pytest.raises(ValueError, match="does not populate"):
        adapter.validate_config(
            EvaluatorConfig(
                task_mode="generalization",
                train_split="train",
                val_split="eval_in_distribution",
                final_split="eval_out_of_distribution",
            )
        )


@pytest.mark.parametrize(
    ("split", "source"),
    (
        ("train", "train_split"),
        ("eval_in_distribution", "val_split"),
        ("eval_out_of_distribution", "final_split"),
    ),
)
def test_resolve_alma_split_accepts_supported_values(split, source):
    assert resolve_alma_split(split, source=source) == split


def test_resolve_alma_split_rejects_unsupported_values():
    with pytest.raises(ValueError, match="Unsupported ALMA split 'val'"):
        resolve_alma_split("val")


def test_summarize_feedback_handles_alfworld_payload_shapes():
    data = {
        "benchmark_eval_score": {"benchmark_overall_eval_score": 0.61},
        "examples": [
            {
                "obs": "You are in the kitchen beside a closed microwave.",
                "actions_list": ["open microwave", "go to sink"],
                "memory_retrieved": {},
                "final_reward": 0.2,
            },
            {
                "error_info": {"error_type": "RuntimeError", "message": "planner failed"},
                "final_reward": 0.0,
            },
        ],
        "token_usage": {
            "gpt-4o-mini": {
                "total_tokens": 90,
                "prompt_tokens": 55,
                "completion_tokens": 35,
                "completion_tokens_details": {"reasoning_tokens": 7},
            }
        },
    }

    feedback = summarize_feedback(ALMA_ALFWORLD_TASK, data, success=False)

    assert "ALMA ALFWorld score: 0.6100" in feedback
    assert "obs=You are in the kitchen beside a closed microwave." in feedback
    assert "actions_list=open microwave, go to sink" in feedback
    assert "Token summary: prompt=55, completion=35, reasoning=7, total=90, models=1" in feedback


def test_summarize_feedback_handles_minihack_context_fields():
    data = {
        "benchmark_eval_score": {"benchmark_overall_eval_score": 0.44},
        "examples": [
            {
                "goal": "Reach the stairs down.",
                "short_term_context": "You stand next to a locked door.",
                "long_term_context": "A corridor continues east.",
                "memo_retrieved": {"hint": "search for the key"},
                "final_reward": 0.1,
            }
        ],
    }

    feedback = summarize_feedback(ALMA_MINIHACK_TASK, data, success=True)

    assert "ALMA MiniHack score: 0.4400" in feedback
    assert "goal=Reach the stairs down." in feedback
    assert "short_term_context=You stand next to a locked door." in feedback


def test_stage1_maps_smoke_success_and_failure(monkeypatch, tmp_path):
    adapter = make_alma_benchmark_adapter(ALMA_TEXTWORLD_TASK)
    candidate_path = _candidate_file(tmp_path)
    fixture = {
        "benchmark_eval_score": {"benchmark_overall_eval_score": 0.42},
        "examples": [],
    }

    monkeypatch.setattr(
        adapter,
        "_run_alma_candidate",
        lambda code_str, *, mode, split: _async_result(
            (True, fixture, "sequential", "eval_in_distribution")
        ),
    )
    success_result = adapter.evaluate_stage1(str(candidate_path), split="eval_in_distribution")
    assert success_result["combined_score"] == 1.0
    assert success_result["validity"] == 1.0
    assert success_result["artifacts"]["alma_eval_type"] == "sequential"
    assert success_result["artifacts"]["alma_status"] == "eval_in_distribution"
    assert success_result["artifacts"]["alma_update_task"] == "eval_in_distribution"

    monkeypatch.setattr(
        adapter,
        "_run_alma_candidate",
        lambda code_str, *, mode, split: _async_result((False, fixture, "sequential", "train")),
    )
    failure_result = adapter.evaluate_stage1(str(candidate_path))
    assert failure_result["combined_score"] == 0.0
    assert failure_result["validity"] == 0.0
    assert "sampled execution failures observed" in failure_result["artifacts"]["feedback"]


def test_stage2_extracts_score_and_token_metrics(monkeypatch, tmp_path):
    adapter = make_alma_benchmark_adapter(ALMA_TEXTWORLD_TASK)
    candidate_path = _candidate_file(tmp_path)
    fixture = {
        "benchmark_eval_score": {
            "benchmark_overall_eval_score": 0.42,
            "benchmark_overall_eval_standard_deviation": 0.08,
        },
        "examples": [
            {"memo_retrieved": {}, "final_reward": 0.0},
            {
                "error_info": {
                    "error_type": "RuntimeError",
                    "message": "Prompt exceeded model context limit.",
                }
            },
        ],
        "token_usage": {
            "gpt-4o-mini": {
                "total_tokens": 120,
                "prompt_tokens": 70,
                "completion_tokens": 50,
                "completion_tokens_details": {"reasoning_tokens": 11},
            }
        },
    }

    monkeypatch.setattr(
        adapter,
        "_run_alma_candidate",
        lambda code_str, *, mode, split: _async_result(
            (False, fixture, "sequential", "eval_in_distribution")
        ),
    )

    result = adapter.evaluate_stage2(str(candidate_path), split="eval_in_distribution")

    assert result["combined_score"] == pytest.approx(0.42)
    assert result["benchmark_overall_eval_standard_deviation"] == pytest.approx(0.08)
    assert result["alma_total_tokens"] == pytest.approx(120.0)
    assert result["alma_reasoning_tokens"] == pytest.approx(11.0)
    assert result["sampled_execution_failures"] == pytest.approx(1.0)
    assert result["artifacts"]["alma_eval_type"] == "sequential"
    assert result["artifacts"]["alma_status"] == "eval_in_distribution"
    assert result["artifacts"]["alma_update_task"] == "eval_in_distribution"


@pytest.mark.asyncio
async def test_generalization_id_runs_keep_split_aligned_updates(monkeypatch, tmp_path):
    calls_path = tmp_path / "calls.json"
    alma_root = _write_fake_alma_root(
        tmp_path,
        score_map={
            "train": 0.25,
            "eval_in_distribution": 0.85,
        },
        record_path=calls_path,
    )
    evaluator_path = _bridge_eval_file(tmp_path)
    candidate_path = _candidate_file(tmp_path)

    monkeypatch.setenv("ALMA_ROOT", str(alma_root))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")

    evaluator = create_evaluator(
        EvaluatorConfig(
            evaluation_file=str(evaluator_path),
            task_mode="generalization",
            train_split="train",
            val_split="eval_in_distribution",
            final_split="eval_in_distribution",
            max_retries=0,
        )
    )

    try:
        search_result = await evaluator.evaluate_program(candidate_path.read_text(encoding="utf-8"), "prog-1")
        final_result = await evaluator.evaluate_program(
            candidate_path.read_text(encoding="utf-8"),
            "prog-1",
            mode="test",
        )
    finally:
        evaluator.close()

    assert search_result.metrics["combined_score"] == pytest.approx(0.85)
    assert search_result.metrics["train_combined_score"] == pytest.approx(0.25)
    assert search_result.metrics["val_combined_score"] == pytest.approx(0.85)
    assert "ALMA TextWorld score: 0.2500" in search_result.artifacts["feedback"]

    assert final_result.metrics["combined_score"] == pytest.approx(0.85)
    assert "train_combined_score" not in final_result.metrics

    recorded_calls = json.loads(calls_path.read_text(encoding="utf-8"))
    assert {call["status"] for call in recorded_calls} == {
        "train",
        "eval_in_distribution",
    }
    assert {call["eval_type"] for call in recorded_calls} == {"sequential"}
    assert {call["update_task"] for call in recorded_calls} == {
        "train",
        "eval_in_distribution",
    }
    assert {call["mode"] for call in recorded_calls} == {"test", "eval"}


def test_alfworld_ood_uses_id_warm_start_protocol(monkeypatch, tmp_path):
    calls_path = tmp_path / "calls.json"
    alma_root = _write_fake_alma_root(tmp_path, record_path=calls_path)
    adapter = make_alma_benchmark_adapter(ALMA_ALFWORLD_TASK)
    candidate_path = _candidate_file(tmp_path)

    monkeypatch.setenv("ALMA_ROOT", str(alma_root))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")

    adapter.validate_config(
        EvaluatorConfig(
            task_mode="generalization",
            train_split="train",
            val_split="eval_in_distribution",
            final_split="eval_out_of_distribution",
        )
    )

    result = adapter.evaluate_stage2(str(candidate_path), split="eval_out_of_distribution")

    assert result["artifacts"]["alma_eval_type"] == "sequential"
    assert result["artifacts"]["alma_update_task"] == "eval_in_distribution"

    recorded_calls = json.loads(calls_path.read_text(encoding="utf-8"))
    assert recorded_calls == [
        {
            "task_type": "alfworld",
            "mode": "eval",
            "eval_type": "sequential",
            "status": "eval_out_of_distribution",
            "update_task": "eval_in_distribution",
            "train_size": 30,
            "update_size": 70,
        }
    ]


def test_alfworld_train_runs_use_upstream_train_size(monkeypatch, tmp_path):
    calls_path = tmp_path / "calls.json"
    alma_root = _write_fake_alma_root(tmp_path, record_path=calls_path)
    adapter = make_alma_benchmark_adapter(ALMA_ALFWORLD_TASK)
    candidate_path = _candidate_file(tmp_path)

    monkeypatch.setenv("ALMA_ROOT", str(alma_root))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")

    adapter.evaluate_stage2(str(candidate_path), split="train")

    recorded_calls = json.loads(calls_path.read_text(encoding="utf-8"))
    assert recorded_calls == [
        {
            "task_type": "alfworld",
            "mode": "eval",
            "eval_type": "batched",
            "status": "train",
            "update_task": "train",
            "train_size": 30,
            "update_size": None,
        }
    ]
