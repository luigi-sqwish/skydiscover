"""Split-aware evaluation coordinator.

Wraps a concrete evaluator backend (Python, containerized, or Harbor) and
normalizes split-aware evaluation into the single-metrics shape used by the
search algorithms.
"""

from __future__ import annotations

import asyncio
from typing import Dict, Iterable, Mapping

from skydiscover.config import EvaluatorConfig
from skydiscover.evaluation.evaluation_result import EvaluationResult

SPLIT_ARTIFACTS_KEY = "__split_artifacts__"
EVALUATION_META_KEY = "__evaluation_meta__"
_FALLBACK_EXCLUDED_METRICS = {"error", "timeout", "disk_space_error"}


def is_auxiliary_metric(metric_name: str) -> bool:
    """Return True for non-selection split metrics that should be ignored as new axes."""
    return metric_name.startswith(("train_", "val_", "final_", "test_"))


class SplitEvaluationCoordinator:
    """Coordinate split-aware evaluation on top of one concrete evaluator backend."""

    def __init__(self, evaluator, config: EvaluatorConfig):
        self.evaluator = evaluator
        self.config = config

    def __getattr__(self, name: str):
        """Preserve the wrapped evaluator interface for existing callers."""
        return getattr(self.evaluator, name)

    async def evaluate_program(
        self,
        program_solution: str,
        program_id: str = "",
        mode: str = "train",
        split: str | None = None,
        phase: str | None = None,
    ) -> EvaluationResult:
        resolved_phase = phase or ("final" if mode == "test" else "search")
        requested_splits = [split] if split else self._splits_for_phase(resolved_phase)
        requested_mode = "test" if resolved_phase == "final" else "train"

        split_results: Dict[str, EvaluationResult] = {}
        for split_name in requested_splits:
            split_results[split_name] = await self.evaluator.evaluate_program(
                program_solution,
                program_id,
                mode=requested_mode,
                split=split_name,
                phase=resolved_phase,
            )

        return self._merge_results(split_results, phase=resolved_phase)

    async def evaluate_batch(
        self,
        programs,
        *,
        phase: str = "search",
        split: str | None = None,
    ):
        coros = [
            self.evaluate_program(solution, program_id, phase=phase, split=split)
            for solution, program_id in programs
        ]
        return await asyncio.gather(*coros)

    def close(self) -> None:
        if hasattr(self.evaluator, "close"):
            self.evaluator.close()

    def _splits_for_phase(self, phase: str) -> list[str]:
        if phase == "final":
            return [self.config.resolved_final_split]
        return self.config.search_splits()

    def _selection_split_for_phase(self, phase: str) -> str:
        if phase == "final":
            return self.config.resolved_final_split
        return self.config.selection_split

    def _metric_prefix_for_split(self, split_name: str, phase: str) -> str:
        """Return the stable role-based prefix for a split's bookkeeping metrics."""
        if phase == "final":
            return "final"
        if split_name == self.config.train_split:
            return "train"
        if self.config.val_split and split_name == self.config.val_split:
            return "val"
        return split_name

    def _merge_results(
        self,
        split_results: Mapping[str, EvaluationResult],
        *,
        phase: str,
    ) -> EvaluationResult:
        split_metrics = {
            split_name: _with_combined_score(result.metrics)
            for split_name, result in split_results.items()
        }
        selection_split = self._selection_split_for_phase(phase)
        if selection_split not in split_results:
            selection_split = next(iter(split_results))
        selection_metrics = split_metrics[selection_split]

        metrics = dict(selection_metrics)
        if phase != "final" and self._should_emit_split_aliases():
            for split_name, result_metrics in split_metrics.items():
                metric_prefix = self._metric_prefix_for_split(split_name, phase)
                for key, value in result_metrics.items():
                    metrics[f"{metric_prefix}_{key}"] = value

        artifacts = self._merge_artifacts(
            split_results,
            selection_split=selection_split,
            phase=phase,
        )

        return EvaluationResult(metrics=metrics, artifacts=artifacts)

    def _should_emit_split_aliases(self) -> bool:
        """Preserve the legacy metric shape for plain single-task evaluation."""
        return self.config.task_mode != "single_task"

    def _merge_artifacts(
        self,
        split_results: Mapping[str, EvaluationResult],
        *,
        selection_split: str,
        phase: str,
    ) -> Dict[str, object]:
        selection_result = split_results[selection_split]
        artifacts: Dict[str, object] = dict(selection_result.artifacts or {})

        # In generalization mode, legacy prompt builders still read the top-level
        # feedback key. Keep that pointed at the training diagnostics rather than
        # the validation/selection split.
        if phase != "final" and self.config.task_mode == "generalization":
            train_result = split_results.get(self.config.train_split)
            train_feedback = (
                (train_result.artifacts or {}).get("feedback")
                if train_result is not None
                else None
            )
            if train_feedback is not None:
                artifacts["feedback"] = train_feedback

        split_artifacts = {
            split_name: dict(result.artifacts or {})
            for split_name, result in split_results.items()
            if result.artifacts
        }
        if split_artifacts:
            artifacts[SPLIT_ARTIFACTS_KEY] = split_artifacts

        artifacts[EVALUATION_META_KEY] = {
            "task_mode": self.config.task_mode,
            "selection_split": selection_split,
            "train_split": self.config.train_split,
            "val_split": self.config.val_split,
            "final_split": self.config.resolved_final_split,
            "phase": phase,
        }
        return artifacts


def merge_prefixed_metrics(
    base_metrics: Mapping[str, object],
    new_metrics: Mapping[str, object],
    prefix: str,
    *,
    include_legacy_test_alias: bool = False,
) -> Dict[str, object]:
    """Return a new metrics dict with ``new_metrics`` added under ``prefix``."""
    merged = dict(base_metrics)
    for key, value in new_metrics.items():
        merged[f"{prefix}{key}"] = value
        if include_legacy_test_alias and prefix == "final_":
            merged[f"test_{key}"] = value
    return merged


def _with_combined_score(metrics: Mapping[str, object]) -> Dict[str, object]:
    """Ensure split metrics always expose a stable combined_score for selection."""
    normalized = dict(metrics)
    if "combined_score" not in normalized:
        normalized["combined_score"] = _fallback_combined_score(normalized)
    return normalized


def _fallback_combined_score(metrics: Mapping[str, object]) -> float:
    """Average user-facing numeric metrics when an evaluator omits combined_score."""
    numeric_values = [
        float(value)
        for key, value in metrics.items()
        if key not in _FALLBACK_EXCLUDED_METRICS
        and not isinstance(value, bool)
        and isinstance(value, (int, float))
    ]
    return sum(numeric_values) / len(numeric_values) if numeric_values else 0.0


def merge_prefixed_artifacts(
    base_artifacts: Mapping[str, object],
    new_artifacts: Mapping[str, object],
    *,
    final_split_name: str,
) -> Dict[str, object]:
    """Merge final-phase artifacts into the stored program artifacts."""
    merged = dict(base_artifacts)
    split_artifacts = dict(merged.get(SPLIT_ARTIFACTS_KEY, {}) or {})

    phase_artifacts = {
        key: value
        for key, value in new_artifacts.items()
        if key not in {SPLIT_ARTIFACTS_KEY, EVALUATION_META_KEY}
    }
    if phase_artifacts:
        split_artifacts["final"] = phase_artifacts
        split_artifacts[final_split_name] = phase_artifacts
        merged[SPLIT_ARTIFACTS_KEY] = split_artifacts

    eval_meta = dict(merged.get(EVALUATION_META_KEY, {}) or {})
    eval_meta["final_split"] = final_split_name
    merged[EVALUATION_META_KEY] = eval_meta
    return merged


def iter_user_artifact_sections(artifacts: Mapping[str, object]) -> Iterable[tuple[str, object]]:
    """Yield user-facing artifact sections while skipping internal metadata keys."""
    for key, value in artifacts.items():
        if key in {SPLIT_ARTIFACTS_KEY, EVALUATION_META_KEY}:
            continue
        yield key, value
