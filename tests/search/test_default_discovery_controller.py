"""Regression tests for DiscoveryController initialization hooks."""

import textwrap

from skydiscover.config import Config, DatabaseConfig, LLMConfig, LLMModelConfig
from skydiscover.search.best_of_n.database import BestOfNDatabase
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)


class _DummyLLM:
    pass


def test_agentic_codebase_root_can_be_resolved_from_evaluator(tmp_path):
    evaluator_path = tmp_path / "eval.py"
    evaluator_path.write_text(
        textwrap.dedent(
            """
            def resolve_agentic_codebase_root():
                return "/tmp/fake-alma-root"

            def evaluate(program_path):
                return {"combined_score": 1.0}
            """
        ),
        encoding="utf-8",
    )

    config = Config()
    config.llm = LLMConfig(
        models=[
            LLMModelConfig(
                name="dummy-model",
                api_key="dummy-key",
                init_client=lambda cfg: _DummyLLM(),
            )
        ]
    )
    config.search.type = "best_of_n"
    config.search.database = DatabaseConfig()
    config.agentic.enabled = True
    config.agentic.codebase_root = None

    controller = DiscoveryController(
        DiscoveryControllerInput(
            config=config,
            evaluation_file=str(evaluator_path),
            database=BestOfNDatabase("best_of_n", config.search.database),
        )
    )
    try:
        assert controller.config.agentic.codebase_root == "/tmp/fake-alma-root"
    finally:
        controller.close()
