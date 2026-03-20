"""Helpers for using SkyDiscover evaluators from external backends."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

from skydiscover.config import Config, EvaluatorConfig
from skydiscover.evaluation import create_evaluator


def build_runtime_evaluator_config(
    config_obj: Config,
    evaluation_file: str,
    file_suffix: str,
) -> EvaluatorConfig:
    evaluator_config = deepcopy(config_obj.evaluator)
    evaluator_config.evaluation_file = os.path.abspath(evaluation_file)
    evaluator_config.file_suffix = file_suffix
    evaluator_config.is_image_mode = config_obj.language == "image"
    evaluator_config.validate()
    return evaluator_config


def create_runtime_evaluator(
    config_obj: Config,
    evaluation_file: str,
    file_suffix: str,
    *,
    max_concurrent: int = 4,
):
    evaluator_config = build_runtime_evaluator_config(
        config_obj,
        evaluation_file=evaluation_file,
        file_suffix=file_suffix,
    )
    return create_evaluator(evaluator_config, max_concurrent=max_concurrent)


def write_external_evaluator_bridge(
    *,
    config_obj: Config,
    evaluation_file: str,
    file_suffix: str,
    output_dir: str,
) -> str:
    """Write a Python evaluator wrapper usable by external search libraries."""
    os.makedirs(output_dir, exist_ok=True)
    bridge_path = os.path.join(output_dir, "skydiscover_external_evaluator.py")
    package_parent = str(Path(__file__).resolve().parents[2])

    evaluator_config = build_runtime_evaluator_config(
        config_obj,
        evaluation_file=evaluation_file,
        file_suffix=file_suffix,
    )
    payload = json.dumps(
        {
            "evaluation_file": evaluator_config.evaluation_file,
            "file_suffix": evaluator_config.file_suffix,
            "is_image_mode": evaluator_config.is_image_mode,
            "task_mode": evaluator_config.task_mode,
            "train_split": evaluator_config.train_split,
            "val_split": evaluator_config.val_split,
            "final_split": evaluator_config.final_split,
            "timeout": evaluator_config.timeout,
            "max_retries": evaluator_config.max_retries,
            "cascade_evaluation": evaluator_config.cascade_evaluation,
            "cascade_thresholds": evaluator_config.cascade_thresholds,
            "llm_as_judge": evaluator_config.llm_as_judge,
        }
    )

    wrapper = f"""\
import atexit
import asyncio
import json
import os
import sys

_PACKAGE_PARENT = {package_parent!r}
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

from skydiscover.config import EvaluatorConfig
from skydiscover.evaluation import create_evaluator

_CONFIG = EvaluatorConfig(**json.loads({payload!r}))
_EVALUATOR = create_evaluator(_CONFIG)
atexit.register(lambda: getattr(_EVALUATOR, "close", lambda: None)())


def evaluate(program_path):
    with open(program_path, "r", encoding="utf-8") as f:
        program_solution = f.read()

    phase = os.environ.get("SKYDISCOVER_PHASE")
    if phase is None:
        phase = "final" if os.environ.get("SKYDISCOVER_MODE") == "test" else "search"
    mode = "test" if phase == "final" else "train"

    result = asyncio.run(
        _EVALUATOR.evaluate_program(
            program_solution,
            os.path.basename(program_path),
            mode=mode,
            phase=phase,
        )
    )
    return result.to_dict()
"""

    with open(bridge_path, "w", encoding="utf-8") as f:
        f.write(wrapper)

    return bridge_path
