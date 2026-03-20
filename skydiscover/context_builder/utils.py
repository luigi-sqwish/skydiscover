"""Shared utilities for context builders."""

from pathlib import Path
from typing import Any, Optional

from skydiscover.evaluation.coordinator import (
    EVALUATION_META_KEY,
    SPLIT_ARTIFACTS_KEY,
    iter_user_artifact_sections,
)


class TemplateManager:
    """Loads .txt templates from one or more directories.

    Directories are processed in order; later directories override
    templates with the same name from earlier ones.
    """

    def __init__(self, *directories: Optional[str]):
        """
        Initializes the TemplateManager with the given directories.
        If there are multiple directories, the templates from the later directories will override
        the templates from the earlier directories.
        """
        self.templates: dict[str, str] = {}
        for d in directories:
            if d:
                path = Path(d)
                if path.exists():
                    self._load_from_directory(path)

    def _load_from_directory(self, directory: Path) -> None:
        for txt_file in directory.glob("*.txt"):
            with open(txt_file, "r") as f:
                self.templates[txt_file.stem] = f.read()

    def get_template(self, name: str) -> str:
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        return self.templates[name]


def prog_attr(program: Any, key: str, default: Any = "") -> Any:
    """Read an attribute from a Program object or a plain dict."""
    if hasattr(program, key):
        return getattr(program, key)
    if isinstance(program, dict):
        return program.get(key, default)
    return default


def format_artifacts(program: Any, heading: str = "##", max_len: int = 2000) -> str:
    """Format evaluator artifacts (e.g. feedback) into markdown sections."""
    artifacts = prog_attr(program, "artifacts", None)
    if not artifacts:
        return ""

    split_artifacts = artifacts.get(SPLIT_ARTIFACTS_KEY)
    eval_meta = artifacts.get(EVALUATION_META_KEY, {}) or {}

    if isinstance(split_artifacts, dict):
        sections = []
        task_mode = eval_meta.get("task_mode")
        train_split = eval_meta.get("train_split")
        selection_split = eval_meta.get("selection_split")
        final_split = eval_meta.get("final_split")

        ordered_splits = []
        if task_mode == "generalization" and train_split:
            ordered_splits.append(train_split)
        if selection_split and selection_split not in ordered_splits:
            ordered_splits.append(selection_split)
        if final_split and final_split not in ordered_splits and final_split in split_artifacts:
            ordered_splits.append(final_split)
        for split_name in split_artifacts.keys():
            if split_name not in ordered_splits:
                ordered_splits.append(split_name)

        for split_name in ordered_splits:
            split_values = split_artifacts.get(split_name)
            if not isinstance(split_values, dict) or not split_values:
                continue

            split_title = split_name.replace("_", " ").title()
            if task_mode == "generalization" and split_name == train_split:
                split_title += " Feedback"
            elif split_name == selection_split:
                split_title += " Selection Artifacts"
            elif split_name == "final":
                split_title = "Final Evaluation Artifacts"

            for key, value in iter_user_artifact_sections(split_values):
                if value is None:
                    continue
                text = str(value)
                if len(text) > max_len:
                    text = text[:max_len] + "\n... (truncated)"
                if key == "feedback":
                    sections.append(f"{heading} {split_title}\n{text}")
                else:
                    sections.append(f"{heading} {split_title}: {key}\n{text}")

        if sections:
            return "\n" + "\n\n".join(sections) + "\n"

    sections = []
    for key, value in iter_user_artifact_sections(artifacts):
        if value is None:
            continue
        text = str(value)
        if len(text) > max_len:
            text = text[:max_len] + "\n... (truncated)"
        if key == "feedback":
            sections.append(f"{heading} Evaluator Feedback\n{text}")
        else:
            sections.append(f"{heading} {key}\n{text}")
    if not sections:
        return ""
    return "\n" + "\n\n".join(sections) + "\n"
