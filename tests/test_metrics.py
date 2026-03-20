"""Tests for metric scoring helpers."""

import pytest

from skydiscover.utils.metrics import get_authoritative_score


def test_get_authoritative_score_prefers_final_metrics():
    metrics = {
        "combined_score": 0.8,
        "final_combined_score": 0.95,
        "final_accuracy": 0.95,
    }

    assert get_authoritative_score(metrics) == pytest.approx(0.95)


def test_get_authoritative_score_falls_back_to_prefixed_average():
    metrics = {
        "combined_score": 0.8,
        "final_accuracy": 0.9,
        "final_speed": 0.7,
    }

    assert get_authoritative_score(metrics) == pytest.approx(0.8)


def test_get_authoritative_score_falls_back_to_search_score_without_final_metrics():
    metrics = {"combined_score": 0.8, "train_combined_score": 0.6}

    assert get_authoritative_score(metrics) == pytest.approx(0.8)
