"""Regression tests for the GEPA external backend wrapper."""

import textwrap

import pytest

from skydiscover.config import Config
from skydiscover.extras.external.gepa_backend import _make_gepa_evaluator


@pytest.mark.asyncio
async def test_gepa_evaluator_runs_inside_async_context(tmp_path):
    evaluator_path = tmp_path / "evaluator.py"
    evaluator_path.write_text(
        textwrap.dedent(
            """
            def evaluate(program_path, *, split="train", phase="search"):
                return {
                    "combined_score": 0.7,
                    "artifacts": {"feedback": f"{split}-{phase}"},
                }
            """
        )
    )

    gepa_evaluator, runtime_evaluator = _make_gepa_evaluator(
        Config(),
        str(evaluator_path),
        ".py",
    )
    try:
        score, side_info = gepa_evaluator("print('hi')")
    finally:
        runtime_evaluator.close()

    assert score == pytest.approx(0.7)
    assert side_info["combined_score"] == pytest.approx(0.7)
    assert side_info["feedback"] == "train-search"
