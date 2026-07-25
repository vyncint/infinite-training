"""The 2.x names must keep working, while warning that 3.0.0 removes them.

These tests pin the backward-compatibility contract: code written against the
2.0.0 API must run unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import tensorflow as tf

from infinite_training import InfinityTraining, Target

X = np.array([[0.0, 0.0], [1.0, 1.0]])
Y = np.array([[0.0], [1.0]])


@pytest.fixture
def legacy_paths(tmp_path) -> dict[str, str]:
    """Checkpoint paths using the 2.0.0 keyword names."""
    return {
        "optimize_weight_path": str(tmp_path / "opt.npy"),
        "last_weight_path": str(tmp_path / "last.npy"),
        "optimize_value_path": str(tmp_path / "opt_val.npy"),
        "list_value_path": str(tmp_path / "list_val.npy"),
    }


@pytest.fixture
def model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(2,))
    outputs = tf.keras.layers.Dense(1)(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


class TestLegacyClassName:
    def test_still_constructs(self, model, legacy_paths):
        with pytest.deprecated_call():
            trainer = InfinityTraining(model=model, **legacy_paths)
        assert trainer is not None

    def test_is_an_infinite_trainer(self, model, legacy_paths):
        from infinite_training import InfiniteTrainer

        with pytest.deprecated_call():
            trainer = InfinityTraining(model=model, **legacy_paths)
        assert isinstance(trainer, InfiniteTrainer)


class TestLegacyConstructorKeywords:
    def test_legacy_paths_are_honoured(self, model, legacy_paths):
        with pytest.deprecated_call():
            trainer = InfinityTraining(model=model, **legacy_paths)
        assert trainer.best_weights_path == legacy_paths["optimize_weight_path"]
        assert trainer.last_weights_path == legacy_paths["last_weight_path"]
        assert trainer.best_value_path == legacy_paths["optimize_value_path"]
        assert trainer.value_history_path == legacy_paths["list_value_path"]

    def test_checkpoints_land_at_the_legacy_paths(self, model, legacy_paths):
        import os

        x, y = X, Y
        with pytest.deprecated_call():
            trainer = InfinityTraining(
                model=model, target=Target("loss", True, math.inf), **legacy_paths
            )
        trainer.compile(optimizer="adam", loss="mse")
        trainer.train(x, y, epochs=1, verbose=0)
        for path in legacy_paths.values():
            assert os.path.exists(path)


class TestLegacyAttributes:
    @pytest.mark.parametrize(
        ("legacy", "current"),
        [
            ("optimize_weight", "best_weights"),
            ("last_weight", "last_weights"),
            ("optimize_value", "best_value"),
            ("list_value", "value_history"),
            ("optimize_model", "best_model"),
        ],
    )
    def test_legacy_attribute_proxies_current_one(self, model, legacy_paths, legacy, current):
        with pytest.deprecated_call():
            trainer = InfinityTraining(model=model, **legacy_paths)
        with pytest.deprecated_call():
            legacy_value = getattr(trainer, legacy)
        current_value = getattr(trainer, current)

        if legacy_value is None or current_value is None:
            assert legacy_value is current_value
        else:
            assert repr(legacy_value) == repr(current_value)

    def test_legacy_attribute_assignment_still_works(self, model, legacy_paths):
        with pytest.deprecated_call():
            trainer = InfinityTraining(model=model, **legacy_paths)
        with pytest.deprecated_call():
            trainer.optimize_value = 0.25
        assert trainer.best_value == 0.25


class TestLegacyMethods:
    def test_predict_optimize_delegates_to_predict_best(self, model, legacy_paths):
        x = X
        with pytest.deprecated_call():
            trainer = InfinityTraining(model=model, **legacy_paths)
        trainer.compile(optimizer="adam", loss="mse")

        with pytest.deprecated_call():
            legacy_output, legacy_value = trainer.predict_optimize(x, verbose=0)
        current_output, current_value = trainer.predict_best(x, verbose=0)

        assert legacy_value == current_value
        assert legacy_output.shape == current_output.shape
