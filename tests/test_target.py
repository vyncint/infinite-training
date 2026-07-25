"""Tests for the Target stopping criterion."""

from __future__ import annotations

import math

import pytest

from infinite_training import Target


class TestDefaults:
    def test_defaults_watch_loss_and_minimise(self):
        target = Target()
        assert target.name == "loss"
        assert target.smaller_is_better is True

    def test_unset_target_is_unreachable_when_minimising(self):
        # No target means "run forever": a loss can never fall below -inf.
        assert Target(smaller_is_better=True).target_value == -math.inf

    def test_unset_target_is_unreachable_when_maximising(self):
        assert Target(smaller_is_better=False).target_value == math.inf

    def test_custom_values_are_preserved(self):
        target = Target(name="accuracy", smaller_is_better=False, target_value=0.95)
        assert target.name == "accuracy"
        assert target.smaller_is_better is False
        assert target.target_value == 0.95


class TestFalsyTargetValues:
    """Zero is a legitimate target and must not be treated as 'unset'."""

    @pytest.mark.parametrize("zero", [0, 0.0])
    def test_zero_target_is_kept_when_minimising(self, zero):
        assert Target(smaller_is_better=True, target_value=zero).target_value == 0

    @pytest.mark.parametrize("zero", [0, 0.0])
    def test_zero_target_is_kept_when_maximising(self, zero):
        assert Target(smaller_is_better=False, target_value=zero).target_value == 0

    def test_zero_target_is_actually_reachable(self):
        # Regression: `target_value or default` swallowed 0.0, leaving -inf,
        # so the stop condition could never fire.
        target = Target(smaller_is_better=True, target_value=0.0)
        assert target.is_reached(-0.5) is True


class TestIsImprovement:
    def test_lower_beats_higher_when_minimising(self):
        target = Target(smaller_is_better=True)
        assert target.is_improvement(0.1, 0.2) is True
        assert target.is_improvement(0.2, 0.1) is False

    def test_higher_beats_lower_when_maximising(self):
        target = Target(smaller_is_better=False)
        assert target.is_improvement(0.9, 0.8) is True
        assert target.is_improvement(0.8, 0.9) is False

    def test_equal_values_are_not_an_improvement(self):
        assert Target(smaller_is_better=True).is_improvement(0.5, 0.5) is False
        assert Target(smaller_is_better=False).is_improvement(0.5, 0.5) is False


class TestIsReached:
    def test_minimising_requires_strictly_below_target(self):
        target = Target(smaller_is_better=True, target_value=0.1)
        assert target.is_reached(0.09) is True
        assert target.is_reached(0.1) is False
        assert target.is_reached(0.2) is False

    def test_maximising_requires_strictly_above_target(self):
        target = Target(smaller_is_better=False, target_value=0.9)
        assert target.is_reached(0.91) is True
        assert target.is_reached(0.9) is False
        assert target.is_reached(0.8) is False

    def test_default_target_is_never_reached(self):
        assert Target(smaller_is_better=True).is_reached(-1e308) is False
        assert Target(smaller_is_better=False).is_reached(1e308) is False


class TestWorstPossibleValue:
    def test_starts_from_positive_infinity_when_minimising(self):
        assert Target(smaller_is_better=True).worst_possible_value == math.inf

    def test_starts_from_negative_infinity_when_maximising(self):
        assert Target(smaller_is_better=False).worst_possible_value == -math.inf

    def test_any_first_measurement_improves_on_it(self):
        for smaller in (True, False):
            target = Target(smaller_is_better=smaller)
            assert target.is_improvement(0.5, target.worst_possible_value) is True
