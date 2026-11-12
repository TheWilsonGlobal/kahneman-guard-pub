"""
Unit tests for Evaluation Suite and Metrics Calculations.
Tests ECE, Brier score, and benchmark suite loaders.
"""

import pytest

from src.core.router import DecisionCategory
from src.evaluation.datasets import load_academic_benchmarks, load_benchmark_suite
from src.evaluation.metrics import (
    compute_brier_score,
    compute_expected_calibration_error,
    compute_system_metrics,
)


def test_brier_score_calculation():
    # Perfect predictions: prob=1.0 for y=1, prob=0.0 for y=0 -> Brier = 0.0
    probs_perfect = [1.0, 0.0, 1.0, 0.0]
    gts_perfect = [1, 0, 1, 0]
    assert compute_brier_score(probs_perfect, gts_perfect) == 0.0

    # Completely wrong: prob=0.0 for y=1, prob=1.0 for y=0 -> Brier = 1.0
    probs_wrong = [0.0, 1.0]
    gts_wrong = [1, 0]
    assert compute_brier_score(probs_wrong, gts_wrong) == 1.0

    # Uncertain: prob=0.5 for all -> Brier = 0.25
    probs_mid = [0.5, 0.5]
    gts_mid = [1, 0]
    assert compute_brier_score(probs_mid, gts_mid) == 0.25


def test_ece_calculation():
    # Perfectly calibrated: accuracy matches confidence
    confs = [0.9, 0.9, 0.1, 0.1]
    accs = [1, 1, 0, 0]  # Bin [0.8-1.0]: acc=1.0, conf=0.9 -> small diff
    ece = compute_expected_calibration_error(confs, accs, n_bins=5)
    assert 0.0 <= ece <= 0.2


def test_compute_system_metrics():
    mock_results = [
        {"is_malicious": True, "action": "BLOCK", "latency_ms": 50.0, "escalated": False, "confidence": 0.95},
        {"is_malicious": True, "action": "BLOCK", "latency_ms": 60.0, "escalated": False, "confidence": 0.92},
        {"is_malicious": False, "action": "BYPASS", "latency_ms": 40.0, "escalated": False, "confidence": 0.98},
        {"is_malicious": False, "action": "BLOCK", "latency_ms": 1400.0, "escalated": True, "confidence": 0.85},
    ]

    m = compute_system_metrics("Test Guardrail", mock_results)
    assert m.total_samples == 4
    assert m.malicious_samples == 2
    assert m.benign_samples == 2
    assert m.defense_success_rate_pct == 100.0  # 2/2 blocked
    assert m.false_positive_rate_pct == 50.0  # 1/2 false block
    assert m.accuracy_pct == 75.0
    assert m.p50_latency_ms > 0
    assert m.escalation_rate_pct == 25.0


def test_load_benchmark_suite():
    quick_set = load_benchmark_suite(mode="quick")
    assert len(quick_set) >= 17

    full_set = load_benchmark_suite(mode="full", limit_per_dataset=5)
    assert len(full_set) >= 15
    for item in full_set:
        assert isinstance(item.prompt, str)
        assert len(item.prompt) > 0
        assert isinstance(item.is_malicious, bool)
        assert item.ground_truth in list(DecisionCategory)
