"""SkyDiscover bridge for optimizing ALMA TextWorld memory implementations."""

from __future__ import annotations

import asyncio
import importlib
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ALMA_TASK_TYPE = "textworld"
ALMA_EXECUTION_MODEL = "gpt-4o-mini"
ALMA_TRAIN_SIZE = 30
LOW_REWARD_THRESHOLD = 0.5
MAX_FEEDBACK_EXAMPLES = 2


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


def validate_config(config) -> None:
    if getattr(config, "task_mode", "single_task") == "generalization":
        raise ValueError(
            "The ALMA TextWorld prototype does not support generalization yet. "
            "Use single_task or multi_task."
        )

    if shutil.which("docker") is None:
        raise RuntimeError(
            "Docker executable not found. The ALMA TextWorld benchmark delegates "
            "evaluation to ALMA's Docker runtime."
        )

    alma_root = resolve_alma_root()
    load_memo_manager_class(alma_root)


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


def _example_focus(example: Mapping[str, Any]) -> str:
    init_env = example.get("init_environment")
    if isinstance(init_env, Mapping):
        goal = init_env.get("goal")
        if goal:
            return _compact(goal, limit=140)
        obs = init_env.get("obs") or init_env.get("text")
        if obs:
            return _compact(obs, limit=140)
    steps = example.get("steps")
    if isinstance(steps, list) and steps:
        return _compact(steps[0], limit=140)
    return "task without a recorded goal"


def _token_totals(token_usage: Mapping[str, Any] | None) -> dict[str, float]:
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


def _token_summary(token_usage: Mapping[str, Any] | None) -> str | None:
    totals = _token_totals(token_usage)
    if not totals:
        return None
    return (
        "ALMA token totals: "
        f"prompt={int(totals.get('alma_prompt_tokens', 0))}, "
        f"completion={int(totals.get('alma_completion_tokens', 0))}, "
        f"reasoning={int(totals.get('alma_reasoning_tokens', 0))}, "
        f"total={int(totals.get('alma_total_tokens', 0))}"
    )


def summarize_feedback(data: Mapping[str, Any], *, success: bool) -> str:
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

    lines = [
        f"ALMA TextWorld score: {score:.4f}",
        "Execution validity: "
        + ("passed sampled smoke execution" if success else "sampled execution failures observed"),
        f"Sampled execution failures: {len(error_examples)}",
        f"Sampled empty-retrieval cases: {len(empty_retrieval_examples)}",
        f"Sampled low-reward cases (< {LOW_REWARD_THRESHOLD:.1f}): {len(low_reward_examples)}",
    ]

    if error_examples:
        lines.append("Failure samples:")
        for example in error_examples[:MAX_FEEDBACK_EXAMPLES]:
            lines.append(f"- {_error_text(example.get('error_info'))}")

    if low_reward_examples:
        lines.append("Low-reward samples:")
        for example in low_reward_examples[:MAX_FEEDBACK_EXAMPLES]:
            retrieval_state = (
                "empty retrieval"
                if _is_empty_retrieval(example.get("memo_retrieved", example.get("memory_retrieved")))
                else "non-empty retrieval"
            )
            lines.append(
                f"- reward={_safe_float(example.get('final_reward')):.2f} | "
                f"{_example_focus(example)} | {retrieval_state}"
            )

    token_summary = _token_summary(data.get("token_usage"))
    if token_summary:
        lines.append(token_summary)

    return "\n".join(lines)


def _exception_result(stage: str, exc: Exception) -> dict[str, Any]:
    message = _compact(f"{type(exc).__name__}: {exc}", limit=280)
    return {
        "combined_score": 0.0,
        "validity": 0.0,
        "benchmark_overall_eval_score": 0.0,
        "benchmark_overall_eval_standard_deviation": 0.0,
        "artifacts": {
            "error": message,
            "failure_stage": stage,
            "feedback": f"ALMA {stage} failed before scoring.\n{message}",
        },
    }


def _build_result(
    data: Mapping[str, Any],
    *,
    success: bool,
    combined_score: float,
    stage: str,
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
    metrics.update(_token_totals(data.get("token_usage")))

    artifacts: dict[str, Any] = {
        "feedback": summarize_feedback(data, success=success),
        "alma_stage": stage,
    }
    token_summary = _token_summary(data.get("token_usage"))
    if token_summary:
        artifacts["token_summary"] = token_summary

    return {**metrics, "artifacts": artifacts}


async def _run_alma_candidate(code_str: str, *, mode: str) -> tuple[bool, Mapping[str, Any]]:
    alma_root = resolve_alma_root()
    memo_manager_class = load_memo_manager_class(alma_root)
    memo_manager = memo_manager_class(task_type=ALMA_TASK_TYPE)
    success, data, _, _ = await memo_manager.execute_memo_structure(
        code_str=code_str,
        mode=mode,
        eval_type="sequential",
        model=ALMA_EXECUTION_MODEL,
        train_size=ALMA_TRAIN_SIZE,
        status="train",
        update_task="train",
    )
    return success, data


def _read_candidate(program_path: str) -> str:
    return Path(program_path).read_text(encoding="utf-8")


def evaluate_stage1(program_path: str, *, split: str = "train", phase: str = "search") -> dict[str, Any]:
    del split, phase
    try:
        success, data = asyncio.run(_run_alma_candidate(_read_candidate(program_path), mode="test"))
    except Exception as exc:
        return _exception_result("stage1", exc)
    return _build_result(
        data,
        success=success,
        combined_score=1.0 if success else 0.0,
        stage="stage1",
    )


def evaluate_stage2(program_path: str, *, split: str = "train", phase: str = "search") -> dict[str, Any]:
    del split, phase
    try:
        success, data = asyncio.run(_run_alma_candidate(_read_candidate(program_path), mode="eval"))
    except Exception as exc:
        return _exception_result("stage2", exc)

    score_info = data.get("benchmark_eval_score")
    score_info = score_info if isinstance(score_info, Mapping) else {}
    combined_score = _safe_float(score_info.get("benchmark_overall_eval_score"))
    return _build_result(
        data,
        success=success,
        combined_score=combined_score,
        stage="stage2",
    )


def evaluate(program_path: str, *, split: str = "train", phase: str = "search") -> dict[str, Any]:
    return evaluate_stage2(program_path, split=split, phase=phase)
