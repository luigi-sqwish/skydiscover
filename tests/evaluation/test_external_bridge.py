"""Tests for the generated evaluator bridge used by external backends."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

from skydiscover.config import Config
from skydiscover.evaluation.external_bridge import write_external_evaluator_bridge


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("bridge_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generated_bridge_preserves_split_aware_selection(tmp_path):
    evaluator_path = tmp_path / "split_eval.py"
    evaluator_path.write_text(
        textwrap.dedent(
            """
            def evaluate(program_path, *, split="train", phase="search"):
                score = 0.2 if split == "train" else 0.8
                if phase == "final":
                    score += 0.05
                return {
                    "combined_score": score,
                    "artifacts": {"feedback": f"{split}-{phase}"},
                }
            """
        )
    )
    program_path = tmp_path / "candidate.py"
    program_path.write_text("print('hi')\n")

    config = Config()
    config.evaluator.task_mode = "generalization"
    config.evaluator.val_split = "val"

    bridge_path = write_external_evaluator_bridge(
        config_obj=config,
        evaluation_file=str(evaluator_path),
        file_suffix=".py",
        output_dir=str(tmp_path),
    )
    module = _load_module(bridge_path)
    result = module.evaluate(str(program_path))

    assert result["combined_score"] == 0.8
    assert result["train_combined_score"] == 0.2
    assert result["val_combined_score"] == 0.8
    assert result["artifacts"]["__split_artifacts__"]["train"]["feedback"] == "train-search"


def test_generated_bridge_is_importable_from_temp_output_dir(tmp_path):
    evaluator_path = tmp_path / "eval.py"
    evaluator_path.write_text(
        textwrap.dedent(
            """
            def evaluate(program_path):
                return {"combined_score": 1.0}
            """
        )
    )
    program_path = tmp_path / "candidate.py"
    program_path.write_text("print('hi')\n")

    bridge_path = write_external_evaluator_bridge(
        config_obj=Config(),
        evaluation_file=str(evaluator_path),
        file_suffix=".py",
        output_dir=str(tmp_path),
    )

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import importlib.util
                import json
                import sys

                spec = importlib.util.spec_from_file_location("bridge_module", sys.argv[1])
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)
                print(json.dumps(module.evaluate(sys.argv[2])))
                """
            ),
            bridge_path,
            str(program_path),
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["combined_score"] == 1.0


def test_generated_bridge_resolves_relative_evaluator_path(tmp_path):
    evaluator_path = tmp_path / "eval.py"
    evaluator_path.write_text(
        textwrap.dedent(
            """
            def evaluate(program_path):
                return {"combined_score": 1.0}
            """
        )
    )
    program_path = tmp_path / "candidate.py"
    program_path.write_text("print('hi')\n")

    relative_evaluator_path = os.path.relpath(evaluator_path, Path.cwd())
    bridge_path = write_external_evaluator_bridge(
        config_obj=Config(),
        evaluation_file=relative_evaluator_path,
        file_suffix=".py",
        output_dir=str(tmp_path),
    )

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import importlib.util
                import json
                import sys

                spec = importlib.util.spec_from_file_location("bridge_module", sys.argv[1])
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)
                print(json.dumps(module.evaluate(sys.argv[2])))
                """
            ),
            bridge_path,
            str(program_path),
        ],
        cwd="/",
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["combined_score"] == 1.0
