"""
Utilities for metric scoring and formatting.
"""

from typing import Any, Dict

_AUXILIARY_METRIC_PREFIXES = ("train_", "val_", "final_", "test_")


def is_auxiliary_metric_name(metric_name: str) -> bool:
    """Return True for split-prefixed bookkeeping metrics."""
    return metric_name.startswith(_AUXILIARY_METRIC_PREFIXES)


def get_score(metrics: Dict[str, Any]) -> float:
    """Return combined_score if available, otherwise average of all numeric metric values."""
    if not metrics:
        return 0.0
    if "combined_score" in metrics:
        try:
            return float(metrics["combined_score"])
        except (ValueError, TypeError):
            pass
    numeric_values = [v for v in metrics.values() if isinstance(v, (int, float))]
    return sum(numeric_values) / len(numeric_values) if numeric_values else 0.0


def get_authoritative_score(metrics: Dict[str, Any]) -> float:
    """Prefer final/test metrics for user-facing reporting when they exist."""
    if not metrics:
        return 0.0

    for prefix in ("final_", "test_"):
        prefixed_metrics = {
            key[len(prefix) :]: value for key, value in metrics.items() if key.startswith(prefix)
        }
        if prefixed_metrics:
            return get_score(prefixed_metrics)

    return get_score(metrics)


def format_metrics(metrics: Dict[str, Any]) -> str:
    """Format a metrics dict for logging, handling both numeric and string values."""
    if not metrics:
        return ""

    parts = []
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            try:
                parts.append(f"{name}={value:.4f}")
            except (ValueError, TypeError):
                parts.append(f"{name}={value}")
        else:
            parts.append(f"{name}={value}")

    return ", ".join(parts)


def format_improvement(parent_metrics: Dict[str, Any], child_metrics: Dict[str, Any]) -> str:
    """Format the per-metric delta between parent and child for logging."""
    if not parent_metrics or not child_metrics:
        return ""

    parts = []
    for metric, child_value in child_metrics.items():
        if metric in parent_metrics:
            parent_value = parent_metrics[metric]
            if isinstance(child_value, (int, float)) and isinstance(parent_value, (int, float)):
                try:
                    parts.append(f"{metric}={child_value - parent_value:+.4f}")
                except (ValueError, TypeError):
                    continue

    return ", ".join(parts)
