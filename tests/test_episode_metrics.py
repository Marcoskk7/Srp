"""
Tests for EpisodeMetrics (methods/base_trainer.py).

EpisodeMetrics has no torch/GPU dependency at all — it only uses numpy
internally. These tests run without a GPU.
"""
import sys
import os
import pytest
import numpy as np

# Ensure the generation root is on sys.path so that the methods package is found.
_GEN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _GEN_ROOT not in sys.path:
    sys.path.insert(0, _GEN_ROOT)

# EpisodeMetrics itself does not use GPU; only the BaseTrainer __init__ does.
# We import the class directly from the module to avoid constructing BaseTrainer.
from methods.base_trainer import EpisodeMetrics  # noqa: E402


# ---------------------------------------------------------------------------
# Empty accumulator
# ---------------------------------------------------------------------------

class TestEpisodeMetricsEmpty:

    def test_compute_returns_dict(self):
        em = EpisodeMetrics()
        result = em.compute()
        assert isinstance(result, dict)

    def test_empty_mean_is_zero(self):
        em = EpisodeMetrics()
        assert em.compute()["mean"] == 0.0

    def test_empty_std_is_zero(self):
        em = EpisodeMetrics()
        assert em.compute()["std"] == 0.0

    def test_empty_max_is_zero(self):
        em = EpisodeMetrics()
        assert em.compute()["max"] == 0.0

    def test_empty_min_is_zero(self):
        em = EpisodeMetrics()
        assert em.compute()["min"] == 0.0

    def test_empty_median_is_zero(self):
        em = EpisodeMetrics()
        assert em.compute()["median"] == 0.0

    def test_empty_all_values_is_empty_list(self):
        em = EpisodeMetrics()
        assert em.compute()["all_values"] == []


# ---------------------------------------------------------------------------
# Single value
# ---------------------------------------------------------------------------

class TestEpisodeMetricsSingleValue:

    def setup_method(self):
        self.em = EpisodeMetrics()
        self.em.update(0.75)

    def test_mean_equals_value(self):
        assert self.em.compute()["mean"] == pytest.approx(0.75)

    def test_std_is_zero(self):
        assert self.em.compute()["std"] == pytest.approx(0.0)

    def test_max_equals_value(self):
        assert self.em.compute()["max"] == pytest.approx(0.75)

    def test_min_equals_value(self):
        assert self.em.compute()["min"] == pytest.approx(0.75)

    def test_median_equals_value(self):
        assert self.em.compute()["median"] == pytest.approx(0.75)

    def test_all_values_contains_value(self):
        assert self.em.compute()["all_values"] == [0.75]


# ---------------------------------------------------------------------------
# Multiple values — correctness
# ---------------------------------------------------------------------------

class TestEpisodeMetricsMultipleValues:

    def setup_method(self):
        self.values = [0.6, 0.7, 0.8, 0.9, 1.0]
        self.em = EpisodeMetrics()
        for v in self.values:
            self.em.update(v)

    def test_mean(self):
        expected = np.mean(self.values)
        assert self.em.compute()["mean"] == pytest.approx(expected)

    def test_std(self):
        expected = np.std(self.values)
        assert self.em.compute()["std"] == pytest.approx(expected)

    def test_max(self):
        assert self.em.compute()["max"] == pytest.approx(max(self.values))

    def test_min(self):
        assert self.em.compute()["min"] == pytest.approx(min(self.values))

    def test_median(self):
        expected = np.median(self.values)
        assert self.em.compute()["median"] == pytest.approx(expected)

    def test_all_values_matches_inserted(self):
        assert self.em.compute()["all_values"] == self.values


# ---------------------------------------------------------------------------
# Loss tracking (optional)
# ---------------------------------------------------------------------------

class TestEpisodeMetricsLoss:

    def test_avg_loss_absent_when_no_loss_provided(self):
        em = EpisodeMetrics()
        em.update(0.9)          # no loss argument
        result = em.compute()
        assert "avg_loss" not in result

    def test_avg_loss_present_when_loss_provided(self):
        em = EpisodeMetrics()
        em.update(0.9, loss=0.3)
        result = em.compute()
        assert "avg_loss" in result

    def test_avg_loss_correct_value(self):
        em = EpisodeMetrics()
        losses = [0.3, 0.5, 0.2]
        for acc, loss in zip([0.8, 0.9, 0.85], losses):
            em.update(acc, loss=loss)
        result = em.compute()
        assert result["avg_loss"] == pytest.approx(np.mean(losses))

    def test_mixed_updates_only_tracks_provided_losses(self):
        """Updates without loss should not contribute to avg_loss."""
        em = EpisodeMetrics()
        em.update(0.8, loss=0.4)
        em.update(0.9)           # no loss
        em.update(0.85, loss=0.2)
        result = em.compute()
        # Only two losses were provided
        assert result["avg_loss"] == pytest.approx(np.mean([0.4, 0.2]))


# ---------------------------------------------------------------------------
# Accumulation order preserved
# ---------------------------------------------------------------------------

class TestEpisodeMetricsOrdering:

    def test_all_values_preserves_insertion_order(self):
        em = EpisodeMetrics()
        values = [0.1, 0.9, 0.5, 0.3, 0.7]
        for v in values:
            em.update(v)
        assert em.compute()["all_values"] == values

    def test_multiple_compute_calls_are_consistent(self):
        """compute() is a pure read — calling it twice gives the same result."""
        em = EpisodeMetrics()
        em.update(0.8)
        em.update(0.9)
        first = em.compute()
        second = em.compute()
        assert first["mean"] == second["mean"]
        assert first["std"] == second["std"]


# ---------------------------------------------------------------------------
# Edge cases: extreme values
# ---------------------------------------------------------------------------

class TestEpisodeMetricsEdgeCases:

    def test_zero_accuracy(self):
        em = EpisodeMetrics()
        em.update(0.0)
        result = em.compute()
        assert result["mean"] == pytest.approx(0.0)
        assert result["max"] == pytest.approx(0.0)
        assert result["min"] == pytest.approx(0.0)

    def test_perfect_accuracy(self):
        em = EpisodeMetrics()
        em.update(1.0)
        result = em.compute()
        assert result["mean"] == pytest.approx(1.0)

    def test_large_number_of_values(self):
        em = EpisodeMetrics()
        values = list(np.linspace(0.0, 1.0, 1000))
        for v in values:
            em.update(v)
        result = em.compute()
        assert result["mean"] == pytest.approx(np.mean(values))
        assert len(result["all_values"]) == 1000
