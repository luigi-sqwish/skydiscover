"""Tests for the ALMA TextWorld benchmark bridge."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import textwrap
import types

import pytest

from skydiscover.config import EvaluatorConfig
from skydiscover.evaluation.evaluator import Evaluator


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_EVALUATOR = REPO_ROOT / "benchmarks" / "alma_textworld" / "evaluator.py"
INITIAL_PROGRAM = REPO_ROOT / "benchmarks" / "alma_textworld" / "initial_program.py"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "alma_textworld_eval_result.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("alma_textworld_eval", BENCHMARK_EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_initial_program_module(monkeypatch):
    agents_pkg = types.ModuleType("agents")
    agents_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "agents", agents_pkg)

    memo_mod = types.ModuleType("agents.memo_structure")

    class MemoStructure:
        def __init__(self):
            pass

    class Sub_memo_layer:
        pass

    memo_mod.MemoStructure = MemoStructure
    memo_mod.Sub_memo_layer = Sub_memo_layer
    monkeypatch.setitem(sys.modules, "agents.memo_structure", memo_mod)

    eval_envs_pkg = types.ModuleType("eval_envs")
    eval_envs_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "eval_envs", eval_envs_pkg)

    base_mod = types.ModuleType("eval_envs.base_envs")

    class Basic_Recorder:
        def __init__(self, init, steps=None, reward=0.0):
            self.init = init
            self.steps = steps or []
            self.reward = reward

    base_mod.Basic_Recorder = Basic_Recorder
    monkeypatch.setitem(sys.modules, "eval_envs.base_envs", base_mod)

    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)

    hire_agent_mod = types.ModuleType("utils.hire_agent")

    class Embedding:
        async def get_embedding(self, text):
            return [1.0] if "task-a" in text else [0.0]

        async def get_batch_embeddings(self, texts):
            return [[1.0] for _ in texts]

        @staticmethod
        async def compute_one_to_group_similarity(target, group):
            return [0.0 for _ in group]

    hire_agent_mod.Embedding = Embedding
    monkeypatch.setitem(sys.modules, "utils.hire_agent", hire_agent_mod)

    spec = importlib.util.spec_from_file_location("alma_textworld_initial_program", INITIAL_PROGRAM)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, Basic_Recorder


def _write_fake_alma_root(tmp_path: Path, stage2_score: float = 0.73) -> Path:
    alma_root = tmp_path / "alma"
    (alma_root / "core").mkdir(parents=True)
    (alma_root / "core" / "__init__.py").write_text("", encoding="utf-8")
    (alma_root / "eval_in_container.py").write_text(
        "async def run_evaluation(*args, **kwargs):\n    return None\n",
        encoding="utf-8",
    )
    (alma_root / "core" / "memo_manager.py").write_text(
        textwrap.dedent(
            f"""
            class Memo_Manager:
                def __init__(self, task_type, **kwargs):
                    self.task_type = task_type

                async def execute_memo_structure(self, code_str=None, mode="test", **kwargs):
                    data = {{
                        "benchmark_eval_score": {{
                            "benchmark_overall_eval_score": {stage2_score if stage2_score else 0.0},
                            "benchmark_overall_eval_standard_deviation": 0.05,
                        }},
                        "examples": [
                            {{
                                "init_environment": {{"goal": "put the apple in the fridge"}},
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
                    success = mode == "test"
                    return success, data, "fake-sha", code_str or ""
            """
        ),
        encoding="utf-8",
    )
    return alma_root


def test_resolve_alma_root_prefers_env_override_and_validates_layout(tmp_path):
    module = _load_module()
    alma_root = _write_fake_alma_root(tmp_path)

    resolved = module.resolve_alma_root({"ALMA_ROOT": str(alma_root)})

    assert resolved == alma_root.resolve()


def test_resolve_agentic_codebase_root_tracks_alma_root(monkeypatch, tmp_path):
    module = _load_module()
    alma_root = _write_fake_alma_root(tmp_path)
    monkeypatch.setenv("ALMA_ROOT", str(alma_root))

    assert module.resolve_agentic_codebase_root() == str(alma_root.resolve())


def test_initial_program_returns_empty_retrieval_when_no_match(monkeypatch):
    module, basic_recorder = _load_initial_program_module(monkeypatch)

    async def exercise_memory():
        memory = module.SimilarityMemory()
        await memory.general_update(
            basic_recorder({"goal": "task-a"}, steps=["take apple"], reward=1.0)
        )
        return await memory.general_retrieve(basic_recorder({"goal": "task-b"}))

    retrieved = asyncio.run(exercise_memory())

    assert retrieved == {"retrieved_similar_tasks": {}}


def test_validate_config_rejects_generalization_before_runtime(monkeypatch, tmp_path):
    module = _load_module()
    alma_root = _write_fake_alma_root(tmp_path)
    monkeypatch.setenv("ALMA_ROOT", str(alma_root))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")

    with pytest.raises(ValueError, match="does not support generalization"):
        module.validate_config(EvaluatorConfig(task_mode="generalization", val_split="val"))


def test_evaluate_stage1_maps_smoke_success_and_failure(monkeypatch):
    module = _load_module()
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    candidate_path = BENCHMARK_EVALUATOR.parent / "initial_program.py"

    monkeypatch.setattr(
        module,
        "_run_alma_candidate",
        lambda code_str, mode: _async_result((True, fixture)),
    )
    success_result = module.evaluate_stage1(str(candidate_path))
    assert success_result["combined_score"] == 1.0
    assert success_result["validity"] == 1.0

    monkeypatch.setattr(
        module,
        "_run_alma_candidate",
        lambda code_str, mode: _async_result((False, fixture)),
    )
    failure_result = module.evaluate_stage1(str(candidate_path))
    assert failure_result["combined_score"] == 0.0
    assert failure_result["validity"] == 0.0
    assert "sampled execution failures observed" in failure_result["artifacts"]["feedback"]


def test_evaluate_stage2_extracts_score_and_token_metrics(monkeypatch):
    module = _load_module()
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    candidate_path = BENCHMARK_EVALUATOR.parent / "initial_program.py"

    monkeypatch.setattr(
        module,
        "_run_alma_candidate",
        lambda code_str, mode: _async_result((False, fixture)),
    )

    result = module.evaluate_stage2(str(candidate_path))

    assert result["combined_score"] == pytest.approx(0.42)
    assert result["benchmark_overall_eval_standard_deviation"] == pytest.approx(0.08)
    assert result["alma_total_tokens"] == pytest.approx(120.0)
    assert result["alma_reasoning_tokens"] == pytest.approx(11.0)
    assert result["sampled_execution_failures"] == pytest.approx(1.0)


def test_summarize_feedback_uses_fixture_examples():
    module = _load_module()
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    feedback = module.summarize_feedback(fixture, success=False)

    assert "ALMA TextWorld score: 0.4200" in feedback
    assert "Sampled execution failures: 1" in feedback
    assert "Sampled empty-retrieval cases: 1" in feedback
    assert "put the apple in the fridge" in feedback
    assert "RuntimeError: Prompt exceeded model context limit." in feedback


@pytest.mark.asyncio
async def test_evaluator_integration_with_fake_alma_root(monkeypatch, tmp_path):
    alma_root = _write_fake_alma_root(tmp_path, stage2_score=0.77)
    candidate_path = tmp_path / "candidate.py"
    candidate_path.write_text("class Candidate: pass\n", encoding="utf-8")

    monkeypatch.setenv("ALMA_ROOT", str(alma_root))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker")

    evaluator = Evaluator(
        EvaluatorConfig(
            evaluation_file=str(BENCHMARK_EVALUATOR),
            cascade_evaluation=True,
            cascade_thresholds=[0.5, 0.5],
            max_retries=0,
        )
    )

    try:
        result = await evaluator.evaluate_program(candidate_path.read_text(encoding="utf-8"), "prog-1")
    finally:
        evaluator.close()

    assert result.metrics["combined_score"] == pytest.approx(0.77)
    assert result.metrics["alma_total_tokens"] == pytest.approx(20.0)
    assert "ALMA TextWorld score: 0.7700" in result.artifacts["feedback"]


async def _async_result(value):
    return value
