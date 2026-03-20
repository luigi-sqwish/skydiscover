"""SkyDiscover bridge for optimizing ALMA MiniHack memory implementations."""

from skydiscover.evaluation.alma_bridge import ALMA_MINIHACK_TASK, make_alma_benchmark_adapter


TASK_DESCRIPTOR = ALMA_MINIHACK_TASK
_ADAPTER = make_alma_benchmark_adapter(TASK_DESCRIPTOR, benchmark_file=__file__)


def resolve_agentic_codebase_root() -> str:
    return _ADAPTER.resolve_agentic_codebase_root()


def validate_config(config) -> None:
    _ADAPTER.validate_config(config)


def evaluate_stage1(program_path: str, *, split: str = "train", phase: str = "search") -> dict:
    return _ADAPTER.evaluate_stage1(program_path, split=split, phase=phase)


def evaluate_stage2(program_path: str, *, split: str = "train", phase: str = "search") -> dict:
    return _ADAPTER.evaluate_stage2(program_path, split=split, phase=phase)


def evaluate(program_path: str, *, split: str = "train", phase: str = "search") -> dict:
    return _ADAPTER.evaluate(program_path, split=split, phase=phase)
