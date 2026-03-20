"""Shared adapter for ALMA-backed memory benchmarks."""

from __future__ import annotations

import asyncio
import importlib
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ALMA_EXECUTION_MODEL = "gpt-4o-mini"
ALMA_SUPPORTED_SPLITS = frozenset(("train", "eval_in_distribution", "eval_out_of_distribution"))
LOW_REWARD_THRESHOLD = 0.5
MAX_FEEDBACK_EXAMPLES = 2


@dataclass(frozen=True)
class ALMATaskDescriptor:
    benchmark_name: str
    task_type: str
    prompt_label: str
    default_train_size: int
    default_timeout: int
    feedback_focus_fields: tuple[str, ...]
    populated_splits: tuple[str, ...] = tuple(ALMA_SUPPORTED_SPLITS)
    rollout_type_by_split: Mapping[str, str] = field(
        default_factory=lambda: {
            split: "batched" for split in ALMA_SUPPORTED_SPLITS
        }
    )
    update_split_overrides: Mapping[str, str] = field(default_factory=dict)
    update_size_overrides: Mapping[str, int] = field(default_factory=dict)


ALMA_TEXTWORLD_TASK = ALMATaskDescriptor(
    benchmark_name="alma_textworld",
    task_type="textworld",
    prompt_label="TextWorld",
    default_train_size=30,
    default_timeout=7200,
    feedback_focus_fields=("goal", "obs"),
    populated_splits=("train", "eval_in_distribution"),
    rollout_type_by_split={
        split: "sequential" for split in ALMA_SUPPORTED_SPLITS
    },
)

ALMA_ALFWORLD_TASK = ALMATaskDescriptor(
    benchmark_name="alma_alfworld",
    task_type="alfworld",
    prompt_label="ALFWorld",
    default_train_size=30,
    default_timeout=10800,
    feedback_focus_fields=("obs", "actions_list"),
    populated_splits=tuple(ALMA_SUPPORTED_SPLITS),
    rollout_type_by_split={
        "train": "batched",
        "eval_in_distribution": "batched",
        "eval_out_of_distribution": "sequential",
    },
    update_split_overrides={"eval_out_of_distribution": "eval_in_distribution"},
    update_size_overrides={"eval_out_of_distribution": 70},
)

ALMA_MINIHACK_TASK = ALMATaskDescriptor(
    benchmark_name="alma_minihack",
    task_type="minihack",
    prompt_label="MiniHack",
    default_train_size=30,
    default_timeout=7200,
    feedback_focus_fields=("goal", "short_term_context", "long_term_context"),
    populated_splits=tuple(ALMA_SUPPORTED_SPLITS),
)

ALMA_BABAISAI_TASK = ALMATaskDescriptor(
    benchmark_name="alma_babaisai",
    task_type="babaisai",
    prompt_label="BabaIsAI",
    default_train_size=30,
    default_timeout=7200,
    feedback_focus_fields=("goal", "obs"),
    populated_splits=("train", "eval_in_distribution"),
)


def _benchmark_file(benchmark_file: str | Path | None = None) -> Path:
    return Path(benchmark_file).resolve() if benchmark_file else Path(__file__).resolve()


def _default_alma_root(benchmark_file: str | Path | None = None) -> Path:
    return _benchmark_file(benchmark_file).parents[2].parent / "alma"


def _compact(text: Any, limit: int = 160) -> str:
    raw = " ".join(str(text).split())
    if len(raw) <= limit:
        return raw
    return raw[: max(limit - 3, 1)] + "..."


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found at {path}")


def resolve_alma_root(
    env: Mapping[str, str] | None = None,
    benchmark_file: str | Path | None = None,
) -> Path:
    env = env or os.environ
    raw_root = env.get("ALMA_ROOT")
    alma_root = Path(raw_root).expanduser().resolve() if raw_root else _default_alma_root(
        benchmark_file
    )

    if not alma_root.exists():
        raise FileNotFoundError(
            f"ALMA root not found at {alma_root}. Set ALMA_ROOT or place the alma repo "
            "next to skydiscover."
        )

    _require_file(alma_root / "core" / "memo_manager.py", "ALMA memo manager")
    _require_file(alma_root / "eval_in_container.py", "ALMA container evaluator")
    return alma_root


def _module_belongs_to_root(module_name: str, root: Path) -> bool:
    module = sys.modules.get(module_name)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    module_path = Path(module_file).resolve()
    return module_path == root or root in module_path.parents


def load_memo_manager_class(alma_root: Path):
    alma_root = alma_root.resolve()
    root_str = str(alma_root)

    for module_name in ("core.memo_manager", "core", "eval_in_container"):
        if module_name in sys.modules and not _module_belongs_to_root(module_name, alma_root):
            sys.modules.pop(module_name, None)

    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    importlib.invalidate_caches()
    try:
        module = importlib.import_module("core.memo_manager")
    except Exception as exc:
        raise ImportError(
            f"Failed to import ALMA Memo_Manager from {alma_root}: {exc}"
        ) from exc

    manager_cls = getattr(module, "Memo_Manager", None)
    if manager_cls is None:
        raise ImportError(f"ALMA Memo_Manager not found in {alma_root / 'core' / 'memo_manager.py'}")
    return manager_cls


def resolve_alma_split(split: str | None, *, source: str = "split") -> str:
    resolved = split or "train"
    if resolved not in ALMA_SUPPORTED_SPLITS:
        supported = ", ".join(sorted(ALMA_SUPPORTED_SPLITS))
        raise ValueError(f"Unsupported ALMA {source} '{resolved}'. Use one of: {supported}.")
    return resolved


def _resolve_populated_split(
    descriptor: ALMATaskDescriptor,
    split: str | None,
    *,
    source: str = "split",
) -> str:
    resolved = resolve_alma_split(split, source=source)
    if resolved not in descriptor.populated_splits:
        supported = ", ".join(descriptor.populated_splits)
        raise ValueError(
            f"ALMA {descriptor.prompt_label} does not populate {source} '{resolved}'. "
            f"Use one of: {supported}."
        )
    return resolved


def _resolve_rollout_type(descriptor: ALMATaskDescriptor, split: str) -> str:
    rollout_type = descriptor.rollout_type_by_split.get(split, "batched")
    if rollout_type not in {"batched", "sequential"}:
        raise ValueError(
            f"Unsupported ALMA rollout type '{rollout_type}' for "
            f"{descriptor.prompt_label} split '{split}'."
        )
    return rollout_type


def _is_empty_retrieval(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Mapping):
        return not value or all(_is_empty_retrieval(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return not value or all(_is_empty_retrieval(v) for v in value)
    return False


def _error_text(error_info: Any) -> str:
    if isinstance(error_info, Mapping):
        err_type = error_info.get("error_type")
        message = error_info.get("message")
        if err_type and message:
            return _compact(f"{err_type}: {message}", limit=220)
        if message:
            return _compact(message, limit=220)
    return _compact(error_info, limit=220)


def _format_focus_value(value: Any) -> str:
    if isinstance(value, Mapping):
        pairs = [f"{key}={_compact(item, limit=80)}" for key, item in list(value.items())[:3]]
        return ", ".join(pairs)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = [_compact(item, limit=60) for item in list(value)[:4]]
        suffix = "" if len(value) <= 4 else ", ..."
        return ", ".join(items) + suffix
    return str(value)


def _lookup_example_field(example: Mapping[str, Any], field: str) -> Any:
    candidate_scopes = (
        example,
        example.get("init_environment"),
        example.get("environment"),
        example.get("context"),
        example.get("task_context"),
    )
    for scope in candidate_scopes:
        if isinstance(scope, Mapping) and field in scope:
            value = scope.get(field)
            if value not in (None, "", [], {}, ()):
                return value
    return None


def _example_focus(example: Mapping[str, Any], descriptor: ALMATaskDescriptor) -> str:
    focus_parts: list[str] = []
    for field in descriptor.feedback_focus_fields:
        value = _lookup_example_field(example, field)
        if value is None:
            continue
        focus_parts.append(f"{field}={_compact(_format_focus_value(value), limit=120)}")
        if len(focus_parts) >= 2:
            break

    if focus_parts:
        return " | ".join(focus_parts)

    steps = example.get("steps")
    if isinstance(steps, list) and steps:
        return f"steps={_compact(steps[0], limit=120)}"

    return "task without recorded focus fields"


def token_totals(token_usage: Mapping[str, Any] | None) -> dict[str, float]:
    totals = {
        "alma_prompt_tokens": 0.0,
        "alma_completion_tokens": 0.0,
        "alma_reasoning_tokens": 0.0,
        "alma_total_tokens": 0.0,
        "alma_models_used": 0.0,
    }
    if not isinstance(token_usage, Mapping) or not token_usage:
        return {}

    totals["alma_models_used"] = float(len(token_usage))

    for usage in token_usage.values():
        if not isinstance(usage, Mapping):
            continue
        totals["alma_prompt_tokens"] += _safe_float(usage.get("prompt_tokens"))
        totals["alma_completion_tokens"] += _safe_float(usage.get("completion_tokens"))
        totals["alma_total_tokens"] += _safe_float(usage.get("total_tokens"))
        completion_details = usage.get("completion_tokens_details")
        if isinstance(completion_details, Mapping):
            totals["alma_reasoning_tokens"] += _safe_float(
                completion_details.get("reasoning_tokens")
            )

    return {key: value for key, value in totals.items() if value > 0}


def token_summary(token_usage: Mapping[str, Any] | None) -> str | None:
    totals = token_totals(token_usage)
    if not totals:
        return None
    return (
        "Token summary: "
        f"prompt={int(totals.get('alma_prompt_tokens', 0))}, "
        f"completion={int(totals.get('alma_completion_tokens', 0))}, "
        f"reasoning={int(totals.get('alma_reasoning_tokens', 0))}, "
        f"total={int(totals.get('alma_total_tokens', 0))}, "
        f"models={int(totals.get('alma_models_used', 0))}"
    )


def summarize_feedback(
    descriptor: ALMATaskDescriptor,
    data: Mapping[str, Any],
    *,
    success: bool,
) -> str:
    score_info = data.get("benchmark_eval_score")
    score_info = score_info if isinstance(score_info, Mapping) else {}
    score = _safe_float(score_info.get("benchmark_overall_eval_score"))
    examples = data.get("examples")
    examples = examples if isinstance(examples, list) else []

    error_examples = [example for example in examples if isinstance(example, Mapping) and "error_info" in example]
    trajectory_examples = [
        example for example in examples if isinstance(example, Mapping) and "error_info" not in example
    ]
    empty_retrieval_examples = [
        example
        for example in trajectory_examples
        if _is_empty_retrieval(example.get("memo_retrieved", example.get("memory_retrieved")))
    ]
    low_reward_examples = [
        example
        for example in trajectory_examples
        if _safe_float(example.get("final_reward")) < LOW_REWARD_THRESHOLD
    ]

    totals = token_totals(data.get("token_usage"))
    lines = [
        f"ALMA {descriptor.prompt_label} score: {score:.4f}",
        "Execution validity: "
        + ("passed sampled smoke execution" if success else "sampled execution failures observed"),
        f"Execution failures: {len(error_examples)}",
        f"Empty retrievals: {len(empty_retrieval_examples)}",
        f"Low-reward examples (< {LOW_REWARD_THRESHOLD:.1f}): {len(low_reward_examples)}",
    ]

    if totals:
        lines.append(
            "Token totals: "
            f"total={int(totals.get('alma_total_tokens', 0))}, "
            f"models={int(totals.get('alma_models_used', 0))}"
        )
        summary = token_summary(data.get("token_usage"))
        if summary:
            lines.append(summary)

    if error_examples:
        lines.append("Failure examples:")
        for example in error_examples[:MAX_FEEDBACK_EXAMPLES]:
            lines.append(f"- {_error_text(example.get('error_info'))}")

    if low_reward_examples:
        lines.append("Low-reward examples:")
        for example in low_reward_examples[:MAX_FEEDBACK_EXAMPLES]:
            retrieval_state = (
                "empty retrieval"
                if _is_empty_retrieval(example.get("memo_retrieved", example.get("memory_retrieved")))
                else "non-empty retrieval"
            )
            lines.append(
                f"- reward={_safe_float(example.get('final_reward')):.2f} | "
                f"{_example_focus(example, descriptor)} | {retrieval_state}"
            )

    return "\n".join(lines)


def _exception_result(descriptor: ALMATaskDescriptor, stage: str, exc: Exception) -> dict[str, Any]:
    message = _compact(f"{type(exc).__name__}: {exc}", limit=280)
    return {
        "combined_score": 0.0,
        "validity": 0.0,
        "benchmark_overall_eval_score": 0.0,
        "benchmark_overall_eval_standard_deviation": 0.0,
        "artifacts": {
            "error": message,
            "failure_stage": stage,
            "feedback": f"ALMA {descriptor.prompt_label} {stage} failed before scoring.\n{message}",
        },
    }


def _build_result(
    descriptor: ALMATaskDescriptor,
    data: Mapping[str, Any],
    *,
    success: bool,
    combined_score: float,
    stage: str,
    scored_split: str,
    eval_type: str,
    update_task: str,
) -> dict[str, Any]:
    score_info = data.get("benchmark_eval_score")
    score_info = score_info if isinstance(score_info, Mapping) else {}
    examples = data.get("examples")
    examples = examples if isinstance(examples, list) else []

    error_count = sum(
        1 for example in examples if isinstance(example, Mapping) and "error_info" in example
    )
    empty_retrieval_count = sum(
        1
        for example in examples
        if isinstance(example, Mapping)
        and "error_info" not in example
        and _is_empty_retrieval(example.get("memo_retrieved", example.get("memory_retrieved")))
    )
    low_reward_count = sum(
        1
        for example in examples
        if isinstance(example, Mapping)
        and "error_info" not in example
        and _safe_float(example.get("final_reward")) < LOW_REWARD_THRESHOLD
    )

    metrics = {
        "combined_score": combined_score,
        "validity": 1.0 if success else 0.0,
        "benchmark_overall_eval_score": _safe_float(
            score_info.get("benchmark_overall_eval_score")
        ),
        "benchmark_overall_eval_standard_deviation": _safe_float(
            score_info.get("benchmark_overall_eval_standard_deviation")
        ),
        "sampled_execution_failures": float(error_count),
        "sampled_empty_retrieval_examples": float(empty_retrieval_count),
        "sampled_low_reward_examples": float(low_reward_count),
    }
    metrics.update(token_totals(data.get("token_usage")))

    artifacts: dict[str, Any] = {
        "feedback": summarize_feedback(descriptor, data, success=success),
        "alma_stage": stage,
        "alma_task_type": descriptor.task_type,
        "alma_status": scored_split,
        "alma_eval_type": eval_type,
        "alma_update_task": update_task,
    }
    summary = token_summary(data.get("token_usage"))
    if summary:
        artifacts["token_summary"] = summary

    return {**metrics, "artifacts": artifacts}


class ALMABenchmarkAdapter:
    """Stateful wrapper that maps SkyDiscover evaluation calls onto ALMA."""

    def __init__(
        self,
        descriptor: ALMATaskDescriptor,
        *,
        benchmark_file: str | Path | None = None,
    ):
        self.descriptor = descriptor
        self.benchmark_file = _benchmark_file(benchmark_file)

    def resolve_agentic_codebase_root(self) -> str:
        return str(resolve_alma_root(benchmark_file=self.benchmark_file))

    def validate_config(self, config) -> None:
        _resolve_populated_split(
            self.descriptor,
            getattr(config, "train_split", "train"),
            source="evaluator.train_split",
        )

        for split_name in self.descriptor.populated_splits:
            _resolve_rollout_type(self.descriptor, split_name)

        for split_name, update_split in self.descriptor.update_split_overrides.items():
            _resolve_populated_split(
                self.descriptor,
                split_name,
                source="split with update split override",
            )
            _resolve_populated_split(
                self.descriptor,
                update_split,
                source=f"update split override for {split_name}",
            )

        for split_name, update_size in self.descriptor.update_size_overrides.items():
            _resolve_populated_split(
                self.descriptor,
                split_name,
                source="split with update size override",
            )
            if not isinstance(update_size, int) or isinstance(update_size, bool) or update_size <= 0:
                raise ValueError(
                    f"Update size override for {split_name} must be a positive integer."
                )

        val_split = getattr(config, "val_split", None)
        if val_split:
            _resolve_populated_split(self.descriptor, val_split, source="evaluator.val_split")

        final_split = getattr(config, "final_split", None)
        if final_split:
            _resolve_populated_split(self.descriptor, final_split, source="evaluator.final_split")

        if shutil.which("docker") is None:
            raise RuntimeError(
                f"Docker executable not found. The ALMA {self.descriptor.prompt_label} benchmark "
                "delegates evaluation to ALMA's Docker runtime."
            )

        alma_root = resolve_alma_root(benchmark_file=self.benchmark_file)
        load_memo_manager_class(alma_root)

    def _read_candidate(self, program_path: str) -> str:
        return Path(program_path).read_text(encoding="utf-8")

    def _resolve_update_split(self, scored_split: str) -> str:
        override = self.descriptor.update_split_overrides.get(scored_split)
        if override is None:
            return scored_split
        return _resolve_populated_split(
            self.descriptor,
            override,
            source=f"update split override for {scored_split}",
        )

    def _resolve_update_size(self, scored_split: str) -> int | None:
        return self.descriptor.update_size_overrides.get(scored_split)

    async def _run_alma_candidate(
        self,
        code_str: str,
        *,
        mode: str,
        split: str = "train",
    ) -> tuple[bool, Mapping[str, Any], str, str]:
        alma_root = resolve_alma_root(benchmark_file=self.benchmark_file)
        memo_manager_class = load_memo_manager_class(alma_root)
        memo_manager = memo_manager_class(task_type=self.descriptor.task_type)
        alma_split = _resolve_populated_split(self.descriptor, split)
        rollout_type = _resolve_rollout_type(self.descriptor, alma_split)
        update_split = self._resolve_update_split(alma_split)
        update_size = self._resolve_update_size(alma_split)
        success, data, _, _ = await memo_manager.execute_memo_structure(
            code_str=code_str,
            mode=mode,
            eval_type=rollout_type,
            model=ALMA_EXECUTION_MODEL,
            train_size=self.descriptor.default_train_size,
            status=alma_split,
            update_task=update_split,
            update_size=update_size,
        )
        return success, data, rollout_type, update_split

    def evaluate_stage1(
        self,
        program_path: str,
        *,
        split: str = "train",
        phase: str = "search",
    ) -> dict[str, Any]:
        del phase
        scored_split = _resolve_populated_split(self.descriptor, split)
        try:
            success, data, rollout_type, update_split = asyncio.run(
                self._run_alma_candidate(
                    self._read_candidate(program_path),
                    mode="test",
                    split=scored_split,
                )
            )
        except Exception as exc:
            return _exception_result(self.descriptor, "stage1", exc)
        return _build_result(
            self.descriptor,
            data,
            success=success,
            combined_score=1.0 if success else 0.0,
            stage="stage1",
            scored_split=scored_split,
            eval_type=rollout_type,
            update_task=update_split,
        )

    def evaluate_stage2(
        self,
        program_path: str,
        *,
        split: str = "train",
        phase: str = "search",
    ) -> dict[str, Any]:
        del phase
        scored_split = _resolve_populated_split(self.descriptor, split)
        try:
            success, data, rollout_type, update_split = asyncio.run(
                self._run_alma_candidate(
                    self._read_candidate(program_path),
                    mode="eval",
                    split=scored_split,
                )
            )
        except Exception as exc:
            return _exception_result(self.descriptor, "stage2", exc)

        score_info = data.get("benchmark_eval_score")
        score_info = score_info if isinstance(score_info, Mapping) else {}
        combined_score = _safe_float(score_info.get("benchmark_overall_eval_score"))
        return _build_result(
            self.descriptor,
            data,
            success=success,
            combined_score=combined_score,
            stage="stage2",
            scored_split=scored_split,
            eval_type=rollout_type,
            update_task=update_split,
        )

    def evaluate(
        self,
        program_path: str,
        *,
        split: str = "train",
        phase: str = "search",
    ) -> dict[str, Any]:
        return self.evaluate_stage2(program_path, split=split, phase=phase)


def make_alma_benchmark_adapter(
    descriptor: ALMATaskDescriptor,
    *,
    benchmark_file: str | Path | None = None,
) -> ALMABenchmarkAdapter:
    return ALMABenchmarkAdapter(descriptor, benchmark_file=benchmark_file)


__all__ = [
    "ALMA_ALFWORLD_TASK",
    "ALMA_BABAISAI_TASK",
    "ALMA_EXECUTION_MODEL",
    "ALMA_MINIHACK_TASK",
    "ALMA_SUPPORTED_SPLITS",
    "ALMA_TEXTWORLD_TASK",
    "ALMATaskDescriptor",
    "ALMABenchmarkAdapter",
    "LOW_REWARD_THRESHOLD",
    "MAX_FEEDBACK_EXAMPLES",
    "load_memo_manager_class",
    "make_alma_benchmark_adapter",
    "resolve_alma_root",
    "resolve_alma_split",
    "summarize_feedback",
    "token_summary",
    "token_totals",
]
