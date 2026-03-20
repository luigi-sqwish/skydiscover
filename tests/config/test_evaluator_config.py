"""Tests for evaluator config defaults and validation."""

import pytest

from skydiscover.config import Config, EvaluatorConfig, apply_overrides, load_config


class TestEvaluatorConfigDefaults:
    def test_default_timeout(self):
        assert EvaluatorConfig().timeout == 360

    def test_default_max_retries(self):
        assert EvaluatorConfig().max_retries == 3

    def test_default_task_mode(self):
        cfg = EvaluatorConfig()
        assert cfg.task_mode == "single_task"
        assert cfg.train_split == "train"
        assert cfg.selection_split == "train"
        assert cfg.resolved_final_split == "train"


class TestEvaluatorSplitValidation:
    def test_generalization_can_be_completed_by_overrides(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("evaluator:\n  task_mode: generalization\n")

        cfg = load_config(str(config_path))
        apply_overrides(cfg, val_split="validation")

        assert cfg.evaluator.task_mode == "generalization"
        assert cfg.evaluator.val_split == "validation"

    def test_generalization_selection_split_is_val(self):
        cfg = EvaluatorConfig(
            task_mode="generalization",
            train_split="train",
            val_split="validation",
        )
        assert cfg.selection_split == "validation"
        assert cfg.search_splits() == ["train", "validation"]

    def test_final_split_defaults_to_selection_split(self):
        cfg = EvaluatorConfig(task_mode="multi_task", train_split="dev")
        assert cfg.resolved_final_split == "dev"


class TestEvaluatorOverrides:
    def test_apply_overrides_updates_split_fields(self):
        cfg = Config()
        apply_overrides(
            cfg,
            task_mode="generalization",
            train_split="train_a",
            val_split="holdout_b",
            final_split="final_c",
        )
        assert cfg.evaluator.task_mode == "generalization"
        assert cfg.evaluator.train_split == "train_a"
        assert cfg.evaluator.val_split == "holdout_b"
        assert cfg.evaluator.final_split == "final_c"

    def test_apply_overrides_rejects_incomplete_generalization(self):
        cfg = Config()
        with pytest.raises(ValueError, match="val_split"):
            apply_overrides(cfg, task_mode="generalization")
