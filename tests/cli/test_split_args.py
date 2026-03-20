"""Tests for split-aware CLI arguments."""

from skydiscover.cli import parse_args


def test_parse_split_aware_args(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "skydiscover-run",
            "initial.py",
            "evaluator.py",
            "--task-mode",
            "generalization",
            "--train-split",
            "train_a",
            "--val-split",
            "val_b",
            "--final-split",
            "test_c",
        ],
    )

    args = parse_args()

    assert args.task_mode == "generalization"
    assert args.train_split == "train_a"
    assert args.val_split == "val_b"
    assert args.final_split == "test_c"
