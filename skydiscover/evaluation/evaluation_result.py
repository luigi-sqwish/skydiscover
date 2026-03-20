import json
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class EvaluationResult:
    """
    Result of program evaluation containing both metrics and optional artifacts
    """

    metrics: Dict[str, float]
    artifacts: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        metrics: Dict[str, float] = {}
        artifacts: Dict[str, Any] = {}

        for key, value in data.items():
            if key == "artifacts" and isinstance(value, dict):
                artifacts.update(value)
            elif isinstance(value, bool):
                metrics[key] = float(value)
            elif isinstance(value, (int, float)):
                metrics[key] = float(value)
            elif value is not None:
                if isinstance(value, (dict, list)):
                    artifacts[key] = json.dumps(value)
                else:
                    artifacts[key] = value

        return cls(metrics=metrics, artifacts=artifacts)

    def to_dict(self) -> Dict[str, Any]:
        result = dict(self.metrics)
        if self.artifacts:
            result["artifacts"] = self.artifacts
        return result
