"""Tests for ALMA benchmark wrappers and configs."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from skydiscover.config import EvaluatorConfig, load_config
from skydiscover.evaluation import create_evaluator


REPO_ROOT = Path(__file__).resolve().parents[2]

BENCHMARKS = (
    {
        "name": "alma_textworld",
        "task_type": "textworld",
        "label": "TextWorld",
        "timeout": 7200,
        "train_size": 30,
        "final_split": "eval_in_distribution",
    },
    {
        "name": "alma_alfworld",
        "task_type": "alfworld",
        "label": "ALFWorld",
        "timeout": 10800,
        "train_size": 30,
        "final_split": "eval_out_of_distribution",
    },
    {
        "name": "alma_minihack",
        "task_type": "minihack",
        "label": "MiniHack",
        "timeout": 7200,
        "train_size": 30,
        "final_split": "eval_out_of_distribution",
    },
    {
        "name": "alma_babaisai",
        "task_type": "babaisai",
        "label": "BabaIsAI",
        "timeout": 7200,
        "train_size": 30,
        "final_split": "eval_in_distribution",
    },
)


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _stub_initial_program_imports(monkeypatch):
    agents_module = types.ModuleType("agents")
    memo_structure_module = types.ModuleType("agents.memo_structure")

    class MemoStructure:
        pass

    class Sub_memo_layer:
        pass

    memo_structure_module.MemoStructure = MemoStructure
    memo_structure_module.Sub_memo_layer = Sub_memo_layer

    eval_envs_module = types.ModuleType("eval_envs")
    base_envs_module = types.ModuleType("eval_envs.base_envs")

    class Basic_Recorder:
        pass

    base_envs_module.Basic_Recorder = Basic_Recorder

    monkeypatch.setitem(sys.modules, "agents", agents_module)
    monkeypatch.setitem(sys.modules, "agents.memo_structure", memo_structure_module)
    monkeypatch.setitem(sys.modules, "eval_envs", eval_envs_module)
    monkeypatch.setitem(sys.modules, "eval_envs.base_envs", base_envs_module)


def _write_fake_alma_root(
    base_dir: Path,
    *,
    score_map: dict[str, float] | None = None,
) -> Path:
    alma_root = base_dir / "alma"
    (alma_root / "core").mkdir(parents=True)
    (alma_root / "core" / "__init__.py").write_text("", encoding="utf-8")
    (alma_root / "eval_in_container.py").write_text(
        "async def run_evaluation(*args, **kwargs):\n    return None\n",
        encoding="utf-8",
    )
    score_map_json = json.dumps(
        score_map
        or {
            "train": 0.77,
            "eval_in_distribution": 0.82,
            "eval_out_of_distribution": 0.61,
        }
    )
    (alma_root / "core" / "memo_manager.py").write_text(
        textwrap.dedent(
            f"""
            import json

            SCORE_MAP = json.loads({score_map_json!r})

            class Memo_Manager:
                def __init__(self, task_type, **kwargs):
                    self.task_type = task_type

                async def execute_memo_structure(self, **kwargs):
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
                            }}
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


@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=[b["name"] for b in BENCHMARKS])
def test_initial_programs_share_same_seed_implementation(benchmark):
    baseline = (
        REPO_ROOT / "benchmarks" / BENCHMARKS[0]["name"] / "initial_program.py"
    ).read_text(encoding="utf-8").rstrip("\n")
    initial_program = (
        REPO_ROOT / "benchmarks" / benchmark["name"] / "initial_program.py"
    ).read_text(encoding="utf-8").rstrip("\n")

    assert initial_program == baseline


@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=[b["name"] for b in BENCHMARKS])
def test_initial_program_handles_empty_retrieval(monkeypatch, benchmark):
    _stub_initial_program_imports(monkeypatch)
    module = _load_module(REPO_ROOT / "benchmarks" / benchmark["name"] / "initial_program.py")
    memory = module.SimilarityMemory()

    retrieved = asyncio.run(
        memory.general_retrieve(
            SimpleNamespace(
                init={"goal": "put the apple in the fridge", "obs": "Kitchen with a closed fridge."},
                steps=[],
                reward=0.0,
            )
        )
    )

    assert retrieved == {"retrieved_similar_tasks": {}}


@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=[b["name"] for b in BENCHMARKS])
def test_initial_program_handles_missing_similarity_match(monkeypatch, benchmark):
    _stub_initial_program_imports(monkeypatch)
    module = _load_module(REPO_ROOT / "benchmarks" / benchmark["name"] / "initial_program.py")
    memory = module.SimilarityMemory()
    memory.sim_db["<goal>heat soup</goal>"] = {"reward": 1.0}

    retrieved = asyncio.run(
        memory.general_retrieve(
            SimpleNamespace(
                init={"goal": "put the apple in the fridge", "obs": "Kitchen with a closed fridge."},
                steps=[],
                reward=0.0,
            )
        )
    )

    assert retrieved == {"retrieved_similar_tasks": {}}


@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=[b["name"] for b in BENCHMARKS])
def test_wrapper_exposes_expected_task_descriptor(benchmark):
    module = _load_module(REPO_ROOT / "benchmarks" / benchmark["name"] / "evaluator.py")

    assert module.TASK_DESCRIPTOR.task_type == benchmark["task_type"]
    assert module.TASK_DESCRIPTOR.prompt_label == benchmark["label"]
    assert module.TASK_DESCRIPTOR.default_timeout == benchmark["timeout"]
    assert module.TASK_DESCRIPTOR.default_train_size == benchmark["train_size"]
    assert module._ADAPTER.descriptor == module.TASK_DESCRIPTOR


@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=[b["name"] for b in BENCHMARKS])
def test_config_defaults_match_plan(benchmark):
    config = load_config(
        str(REPO_ROOT / "benchmarks" / benchmark["name"] / "config_adaevolve.yaml")
    )

    assert config.evaluator.task_mode == "single_task"
    assert config.evaluator.train_split == "train"
    assert config.evaluator.val_split == "eval_in_distribution"
    assert config.evaluator.final_split == benchmark["final_split"]
    assert config.evaluator.timeout == benchmark["timeout"]
    assert config.max_parallel_iterations == 1
    assert config.diff_based_generation is True
    assert config.agentic.enabled is True
    assert config.agentic.codebase_root is None


@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=[b["name"] for b in BENCHMARKS])
def test_wrapper_validate_config_accepts_all_split_modes(monkeypatch, tmp_path, benchmark):
    module = _load_module(REPO_ROOT / "benchmarks" / benchmark["name"] / "evaluator.py")
    alma_root = _write_fake_alma_root(tmp_path)

    monkeypatch.setenv("ALMA_ROOT", str(alma_root))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")

    module.validate_config(
        EvaluatorConfig(
            task_mode="single_task",
            train_split="train",
            val_split="eval_in_distribution",
            final_split=benchmark["final_split"],
        )
    )
    module.validate_config(
        EvaluatorConfig(
            task_mode="multi_task",
            train_split="train",
            val_split="eval_in_distribution",
            final_split=benchmark["final_split"],
        )
    )
    module.validate_config(
        EvaluatorConfig(
            task_mode="generalization",
            train_split="train",
            val_split="eval_in_distribution",
            final_split=benchmark["final_split"],
        )
    )


@pytest.mark.parametrize("benchmark", BENCHMARKS, ids=[b["name"] for b in BENCHMARKS])
@pytest.mark.asyncio
async def test_smoke_evaluator_run_for_each_benchmark(monkeypatch, tmp_path, benchmark):
    alma_root = _write_fake_alma_root(tmp_path)
    benchmark_dir = REPO_ROOT / "benchmarks" / benchmark["name"]
    candidate_text = (benchmark_dir / "initial_program.py").read_text(encoding="utf-8")

    monkeypatch.setenv("ALMA_ROOT", str(alma_root))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")

    evaluator = create_evaluator(
        EvaluatorConfig(
            evaluation_file=str(benchmark_dir / "evaluator.py"),
            task_mode="single_task",
            train_split="train",
            val_split="eval_in_distribution",
            final_split=benchmark["final_split"],
            timeout=benchmark["timeout"],
            max_retries=0,
        )
    )

    try:
        result = await evaluator.evaluate_program(candidate_text, f"{benchmark['name']}-prog")
    finally:
        evaluator.close()

    assert result.metrics["combined_score"] == pytest.approx(0.77)
    assert result.metrics["alma_total_tokens"] == pytest.approx(20.0)
    assert result.artifacts["alma_task_type"] == benchmark["task_type"]
    assert f"ALMA {benchmark['label']} score: 0.7700" in result.artifacts["feedback"]


@pytest.mark.asyncio
async def test_textworld_generalization_uses_val_for_selection(monkeypatch, tmp_path):
    alma_root = _write_fake_alma_root(
        tmp_path,
        score_map={
            "train": 0.31,
            "eval_in_distribution": 0.88,
            "eval_out_of_distribution": 0.57,
        },
    )
    benchmark_dir = REPO_ROOT / "benchmarks" / "alma_textworld"
    candidate_text = (benchmark_dir / "initial_program.py").read_text(encoding="utf-8")

    monkeypatch.setenv("ALMA_ROOT", str(alma_root))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")

    evaluator = create_evaluator(
        EvaluatorConfig(
            evaluation_file=str(benchmark_dir / "evaluator.py"),
            task_mode="generalization",
            train_split="train",
            val_split="eval_in_distribution",
            final_split="eval_in_distribution",
            timeout=7200,
            max_retries=0,
        )
    )

    try:
        search_result = await evaluator.evaluate_program(candidate_text, "textworld-generalization")
        final_result = await evaluator.evaluate_program(
            candidate_text,
            "textworld-generalization",
            mode="test",
        )
    finally:
        evaluator.close()

    assert search_result.metrics["combined_score"] == pytest.approx(0.88)
    assert search_result.metrics["train_combined_score"] == pytest.approx(0.31)
    assert search_result.metrics["val_combined_score"] == pytest.approx(0.88)
    assert "ALMA TextWorld score: 0.3100" in search_result.artifacts["feedback"]

    assert final_result.metrics["combined_score"] == pytest.approx(0.88)
    assert "train_combined_score" not in final_result.metrics
