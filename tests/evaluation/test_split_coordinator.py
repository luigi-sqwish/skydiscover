"""Tests for split-aware evaluation coordination."""

import textwrap

import pytest

from skydiscover.config import EvaluatorConfig
from skydiscover.context_builder.utils import format_artifacts
from skydiscover.evaluation import create_evaluator
from skydiscover.evaluation.coordinator import (
    EVALUATION_META_KEY,
    SPLIT_ARTIFACTS_KEY,
    merge_prefixed_artifacts,
)
from skydiscover.utils.metrics import get_score


def _write_eval_file(tmp_path, content: str):
    path = tmp_path / "evaluator.py"
    path.write_text(textwrap.dedent(content))
    return path


def test_coordinator_preserves_wrapped_evaluator_attributes(tmp_path):
    evaluator_path = _write_eval_file(
        tmp_path,
        """
        def evaluate(program_path):
            return {"combined_score": 0.4}
        """,
    )
    llm_judge = object()

    evaluator = create_evaluator(
        EvaluatorConfig(evaluation_file=str(evaluator_path)),
        llm_judge=llm_judge,
    )

    assert evaluator.llm_judge is llm_judge


def test_create_evaluator_validates_incomplete_generalization_config(tmp_path):
    evaluator_path = _write_eval_file(
        tmp_path,
        """
        def evaluate(program_path):
            return {"combined_score": 0.4}
        """,
    )

    with pytest.raises(ValueError, match="val_split"):
        create_evaluator(
            EvaluatorConfig(
                evaluation_file=str(evaluator_path),
                task_mode="generalization",
            )
        )


@pytest.mark.asyncio
async def test_legacy_python_evaluator_still_works_in_multi_task_mode(tmp_path):
    evaluator_path = _write_eval_file(
        tmp_path,
        """
        def evaluate(program_path):
            return {
                "combined_score": 0.4,
                "feedback": "legacy feedback",
            }
        """,
    )
    evaluator = create_evaluator(
        EvaluatorConfig(
            evaluation_file=str(evaluator_path),
            task_mode="multi_task",
        )
    )

    result = await evaluator.evaluate_program("print('hello')", "prog-1")

    assert result.metrics["combined_score"] == 0.4
    assert result.metrics["train_combined_score"] == 0.4
    assert result.artifacts["feedback"] == "legacy feedback"
    assert result.artifacts[SPLIT_ARTIFACTS_KEY]["train"]["feedback"] == "legacy feedback"


@pytest.mark.asyncio
async def test_single_task_preserves_legacy_metric_shape(tmp_path):
    evaluator_path = _write_eval_file(
        tmp_path,
        """
        def evaluate(program_path):
            return {
                "combined_score": 0.4,
                "accuracy": 0.7,
            }
        """,
    )
    evaluator = create_evaluator(
        EvaluatorConfig(
            evaluation_file=str(evaluator_path),
            task_mode="single_task",
        )
    )

    result = await evaluator.evaluate_program("print('hello')", "prog-single")

    assert result.metrics["combined_score"] == 0.4
    assert result.metrics["accuracy"] == 0.7
    assert "train_combined_score" not in result.metrics
    assert "train_accuracy" not in result.metrics


@pytest.mark.asyncio
async def test_generalization_uses_val_as_selection_split_and_keeps_train_metrics(tmp_path):
    evaluator_path = _write_eval_file(
        tmp_path,
        """
        def evaluate(program_path, *, split="train", phase="search"):
            score_map = {"train": 0.2, "val": 0.9}
            score = score_map[split] + (0.05 if phase == "final" else 0.0)
            return {
                "combined_score": score,
                "artifacts": {"feedback": f"{split}-{phase}-feedback"},
                "notes": f"{split}-{phase}-notes",
            }
        """,
    )
    evaluator = create_evaluator(
        EvaluatorConfig(
            evaluation_file=str(evaluator_path),
            task_mode="generalization",
            train_split="train",
            val_split="val",
        )
    )

    search_result = await evaluator.evaluate_program("print('candidate')", "prog-2")

    assert search_result.metrics["combined_score"] == 0.9
    assert search_result.metrics["train_combined_score"] == 0.2
    assert search_result.metrics["val_combined_score"] == 0.9
    assert search_result.artifacts[EVALUATION_META_KEY]["selection_split"] == "val"
    assert search_result.artifacts[SPLIT_ARTIFACTS_KEY]["train"]["feedback"] == "train-search-feedback"
    assert search_result.artifacts[SPLIT_ARTIFACTS_KEY]["val"]["feedback"] == "val-search-feedback"

    final_result = await evaluator.evaluate_program(
        "print('candidate')",
        "prog-2",
        mode="test",
    )
    assert final_result.metrics["combined_score"] == pytest.approx(0.95)
    assert "train_combined_score" not in final_result.metrics


@pytest.mark.asyncio
async def test_generalization_synthesizes_selection_combined_score_when_missing(tmp_path):
    evaluator_path = _write_eval_file(
        tmp_path,
        """
        def evaluate(program_path, *, split="train", phase="search"):
            return {
                "accuracy": 0.2 if split == "train" else 0.8,
            }
        """,
    )
    evaluator = create_evaluator(
        EvaluatorConfig(
            evaluation_file=str(evaluator_path),
            task_mode="generalization",
            train_split="train",
            val_split="val",
        )
    )

    result = await evaluator.evaluate_program("print('candidate')", "prog-4")

    assert result.metrics["combined_score"] == pytest.approx(0.8)
    assert result.metrics["train_combined_score"] == pytest.approx(0.2)
    assert result.metrics["val_combined_score"] == pytest.approx(0.8)
    assert get_score(result.metrics) == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_generalization_fallback_ignores_timeout_flags(tmp_path):
    evaluator_path = _write_eval_file(
        tmp_path,
        """
        def evaluate(program_path, *, split="train", phase="search"):
            return {
                "timeout": True,
                "error": 0.0,
            }
        """,
    )
    evaluator = create_evaluator(
        EvaluatorConfig(
            evaluation_file=str(evaluator_path),
            task_mode="generalization",
            train_split="train",
            val_split="val",
        )
    )

    result = await evaluator.evaluate_program("print('candidate')", "prog-timeout")

    assert result.metrics["combined_score"] == 0.0
    assert result.metrics["train_combined_score"] == 0.0
    assert result.metrics["val_combined_score"] == 0.0
    assert get_score(result.metrics) == 0.0


@pytest.mark.asyncio
async def test_generalization_uses_canonical_metric_prefixes_for_custom_split_names(tmp_path):
    evaluator_path = _write_eval_file(
        tmp_path,
        """
        def evaluate(program_path, *, split="seen", phase="search"):
            return {
                "accuracy": 0.2 if split == "seen" else 0.8,
            }
        """,
    )
    evaluator = create_evaluator(
        EvaluatorConfig(
            evaluation_file=str(evaluator_path),
            task_mode="generalization",
            train_split="seen",
            val_split="unseen",
        )
    )

    result = await evaluator.evaluate_program("print('candidate')", "prog-custom-splits")

    assert result.metrics["combined_score"] == pytest.approx(0.8)
    assert result.metrics["train_accuracy"] == pytest.approx(0.2)
    assert result.metrics["val_accuracy"] == pytest.approx(0.8)
    assert "seen_accuracy" not in result.metrics
    assert "unseen_accuracy" not in result.metrics


@pytest.mark.asyncio
async def test_generalization_promotes_train_feedback_to_top_level_artifacts(tmp_path):
    evaluator_path = _write_eval_file(
        tmp_path,
        """
        def evaluate(program_path, *, split="train", phase="search"):
            return {
                "combined_score": 0.2 if split == "train" else 0.8,
                "artifacts": {"feedback": f"{split} feedback"},
            }
        """,
    )
    evaluator = create_evaluator(
        EvaluatorConfig(
            evaluation_file=str(evaluator_path),
            task_mode="generalization",
            train_split="train",
            val_split="val",
        )
    )

    result = await evaluator.evaluate_program("print('candidate')", "prog-5")

    assert result.artifacts["feedback"] == "train feedback"
    assert result.artifacts[SPLIT_ARTIFACTS_KEY]["train"]["feedback"] == "train feedback"
    assert result.artifacts[SPLIT_ARTIFACTS_KEY]["val"]["feedback"] == "val feedback"


@pytest.mark.asyncio
async def test_format_artifacts_prefers_train_feedback_in_generalization_mode(tmp_path):
    evaluator_path = _write_eval_file(
        tmp_path,
        """
        def evaluate(program_path, *, split="train", phase="search"):
            score = 0.1 if split == "train" else 0.8
            return {
                "combined_score": score,
                "artifacts": {"feedback": f"{split} diagnostics"},
            }
        """,
    )
    evaluator = create_evaluator(
        EvaluatorConfig(
            evaluation_file=str(evaluator_path),
            task_mode="generalization",
            train_split="train",
            val_split="val",
        )
    )

    result = await evaluator.evaluate_program("print('candidate')", "prog-3")
    rendered = format_artifacts({"artifacts": result.artifacts})

    assert "Train Feedback" in rendered
    assert "train diagnostics" in rendered
    assert "Val Selection Artifacts" in rendered
    assert "val diagnostics" in rendered


def test_merge_prefixed_artifacts_updates_resolved_final_split_alias():
    merged = merge_prefixed_artifacts(
        {
            SPLIT_ARTIFACTS_KEY: {"val": {"feedback": "search diagnostics"}},
            EVALUATION_META_KEY: {"final_split": "val"},
        },
        {"feedback": "final diagnostics"},
        final_split_name="val",
    )

    assert merged[SPLIT_ARTIFACTS_KEY]["final"]["feedback"] == "final diagnostics"
    assert merged[SPLIT_ARTIFACTS_KEY]["val"]["feedback"] == "final diagnostics"
