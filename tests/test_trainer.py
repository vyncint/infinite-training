"""Tests for the training loop, stop conditions and error handling."""

from __future__ import annotations

import math

import numpy as np
import pytest

from infinite_training import InfiniteTrainer, Target


class TestStopConditions:
    def test_stops_once_the_target_is_reached(self, make_trainer, dataset):
        x, y = dataset
        # target_value=inf while minimising is satisfied by any finite loss,
        # so the loop must exit after exactly one round.
        trainer = make_trainer(target=Target("loss", True, math.inf))
        trainer.train(x, y, epochs=1, verbose=0)
        assert trainer.rounds_completed == 1

    def test_stops_once_the_timeout_elapses(self, make_trainer, dataset):
        x, y = dataset
        trainer = make_trainer(target=Target("loss", True), timeout=0)
        trainer.train(x, y, epochs=1, verbose=0)
        # Timeout is checked between rounds, so exactly one round runs.
        assert trainer.rounds_completed == 1

    def test_target_takes_priority_over_timeout(self, make_trainer, dataset):
        x, y = dataset
        trainer = make_trainer(target=Target("loss", True, math.inf), timeout=0)
        trainer.train(x, y, epochs=1, verbose=0)
        assert trainer.rounds_completed == 1


class TestBestTracking:
    def test_records_one_history_entry_per_round(self, make_trainer, dataset):
        x, y = dataset
        trainer = make_trainer(target=Target("loss", True, math.inf))
        trainer.train(x, y, epochs=3, verbose=0)
        # One fit() call == one recorded value, regardless of epochs per call.
        assert trainer.rounds_completed == 1
        assert len(trainer.value_history) == 1

    def test_best_value_matches_the_observed_value(self, make_trainer, dataset):
        x, y = dataset
        trainer = make_trainer(target=Target("loss", True, math.inf))
        trainer.train(x, y, epochs=1, verbose=0)
        assert trainer.best_value == pytest.approx(trainer.last_value)

    def test_best_weights_are_captured(self, make_trainer, dataset):
        x, y = dataset
        trainer = make_trainer(target=Target("loss", True, math.inf))
        trainer.train(x, y, epochs=1, verbose=0)
        assert trainer.best_weights
        assert all(isinstance(w, np.ndarray) for w in trainer.best_weights)

    def test_best_value_is_not_replaced_by_a_worse_round(self, make_trainer, dataset):
        x, y = dataset
        trainer = make_trainer(target=Target("loss", True, math.inf))
        trainer.train(x, y, epochs=1, verbose=0)
        best_after_first = trainer.best_value

        # Force a deliberately worse observation and re-run one round.
        trainer.best_value = -1.0
        trainer.train(x, y, epochs=1, verbose=0)
        assert trainer.best_value == -1.0
        assert best_after_first != -1.0


class TestLastValue:
    def test_is_none_before_training(self, make_trainer):
        # Regression: this used to raise AttributeError because last_value was
        # only ever assigned inside the training loop.
        assert make_trainer().last_value is None

    def test_predict_last_works_before_training(self, make_trainer, dataset):
        x, _ = dataset
        _, value = make_trainer().predict_last(x, verbose=0)
        assert value is None

    def test_reflects_the_most_recent_round(self, make_trainer, dataset):
        x, y = dataset
        trainer = make_trainer(target=Target("loss", True, math.inf))
        trainer.train(x, y, epochs=1, verbose=0)
        assert trainer.last_value == pytest.approx(float(trainer.value_history[-1]))


class TestErrorHandling:
    def test_train_without_compile_is_rejected(self, model, paths, dataset):
        x, y = dataset
        trainer = InfiniteTrainer(model=model, **paths)
        with pytest.raises(RuntimeError, match="compile\\(\\) must be called"):
            trainer.train(x, y, epochs=1, verbose=0)

    def test_inference_without_compile_is_rejected(self, model, paths, dataset):
        x, _ = dataset
        trainer = InfiniteTrainer(model=model, **paths)
        with pytest.raises(RuntimeError, match="compile\\(\\) must be called"):
            trainer.predict_best(x, verbose=0)

    def test_unknown_target_name_explains_what_is_available(self, make_trainer, dataset):
        x, y = dataset
        trainer = make_trainer(target=Target("not_a_metric", True, math.inf))
        with pytest.raises(RuntimeError) as excinfo:
            trainer.train(x, y, epochs=1, verbose=0)
        message = str(excinfo.value)
        assert "not_a_metric" in message
        assert "Available keys" in message
        assert "loss" in message

    def test_validation_target_hints_at_validation_data(self, make_trainer, dataset):
        x, y = dataset
        trainer = make_trainer(target=Target("val_loss", True, math.inf))
        with pytest.raises(RuntimeError, match="validation_data"):
            trainer.train(x, y, epochs=1, verbose=0)

    def test_unexpected_keyword_is_rejected(self, model, paths):
        with pytest.raises(TypeError, match="Unexpected keyword argument"):
            InfiniteTrainer(model=model, not_a_real_option=1, **paths)
