"""Tests for checkpointing and resuming across sessions."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import tensorflow as tf

from infinite_training import InfiniteTrainer, Target


def _fresh_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(2,))
    outputs = tf.keras.layers.Dense(1)(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def _train_once(paths, dataset, **kwargs) -> InfiniteTrainer:
    """Run a single round in a brand-new trainer, as a separate 'session'."""
    x, y = dataset
    trainer = InfiniteTrainer(
        model=_fresh_model(),
        target=Target("loss", True, math.inf),
        **paths,
        **kwargs,
    )
    trainer.compile(optimizer="adam", loss="mse")
    trainer.train(x, y, epochs=1, verbose=0)
    return trainer


class TestCheckpointFiles:
    def test_all_four_checkpoints_are_written(self, paths, dataset):
        _train_once(paths, dataset)
        for path in paths.values():
            assert os.path.exists(path), f"missing checkpoint: {path}"

    def test_parent_directories_are_created(self, tmp_path, dataset):
        nested = {
            "best_weights_path": str(tmp_path / "runs" / "a" / "best.npy"),
            "last_weights_path": str(tmp_path / "runs" / "a" / "last.npy"),
            "best_value_path": str(tmp_path / "runs" / "a" / "value.npy"),
            "value_history_path": str(tmp_path / "runs" / "a" / "history.npy"),
        }
        _train_once(nested, dataset)
        for path in nested.values():
            assert os.path.exists(path)


class TestResume:
    def test_history_accumulates_across_sessions(self, paths, dataset):
        first = _train_once(paths, dataset)
        assert first.rounds_completed == 1

        second = _train_once(paths, dataset)
        # The second session loads the first session's history and appends.
        assert second.rounds_completed == 2

    def test_best_value_is_restored(self, paths, dataset):
        first = _train_once(paths, dataset)
        saved_best = first.best_value

        resumed = InfiniteTrainer(model=_fresh_model(), **paths)
        assert resumed.best_value == pytest.approx(saved_best)
        assert isinstance(resumed.best_value, float)

    def test_weights_are_restored(self, paths, dataset):
        first = _train_once(paths, dataset)

        resumed = InfiniteTrainer(model=_fresh_model(), **paths)
        for restored, original in zip(resumed.last_weights, first.last_weights, strict=True):
            np.testing.assert_allclose(restored, original)
        for restored, original in zip(resumed.best_weights, first.best_weights, strict=True):
            np.testing.assert_allclose(restored, original)

    def test_last_value_survives_a_restart(self, paths, dataset):
        first = _train_once(paths, dataset)

        resumed = InfiniteTrainer(model=_fresh_model(), **paths)
        # Derived from the persisted history, so it is available immediately
        # without re-training.
        assert resumed.last_value == pytest.approx(first.last_value)

    def test_restored_weights_are_loaded_into_the_models(self, paths, dataset):
        _train_once(paths, dataset)

        resumed = InfiniteTrainer(model=_fresh_model(), **paths)
        resumed.compile(optimizer="adam", loss="mse")
        for live, checkpointed in zip(
            resumed.model.get_weights(), resumed.last_weights, strict=True
        ):
            np.testing.assert_allclose(live, checkpointed)
        for live, checkpointed in zip(
            resumed.best_model.get_weights(), resumed.best_weights, strict=True
        ):
            np.testing.assert_allclose(live, checkpointed)


class TestFreshStart:
    def test_absent_checkpoints_fall_back_to_model_state(self, paths):
        model = _fresh_model()
        trainer = InfiniteTrainer(model=model, **paths)
        for loaded, original in zip(trainer.last_weights, model.get_weights(), strict=True):
            np.testing.assert_allclose(loaded, original)

    def test_best_value_starts_at_the_worst_possible(self, paths):
        minimising = InfiniteTrainer(
            model=_fresh_model(), target=Target(smaller_is_better=True), **paths
        )
        assert minimising.best_value == math.inf

    def test_best_value_starts_at_negative_infinity_when_maximising(self, paths):
        maximising = InfiniteTrainer(
            model=_fresh_model(), target=Target(smaller_is_better=False), **paths
        )
        assert maximising.best_value == -math.inf

    def test_history_starts_empty(self, paths):
        trainer = InfiniteTrainer(model=_fresh_model(), **paths)
        assert trainer.rounds_completed == 0
        assert len(trainer.value_history) == 0
