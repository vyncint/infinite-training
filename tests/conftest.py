"""Shared fixtures for the test suite."""

from __future__ import annotations

import os

# Keep TensorFlow quiet; must be set before the first import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import tensorflow as tf  # noqa: E402

from infinite_training import InfiniteTrainer  # noqa: E402


@pytest.fixture
def model() -> tf.keras.Model:
    """A two-input, one-output functional model small enough to fit instantly."""
    inputs = tf.keras.Input(shape=(2,))
    outputs = tf.keras.layers.Dense(1)(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


@pytest.fixture
def dataset() -> tuple[np.ndarray, np.ndarray]:
    """A tiny deterministic regression dataset."""
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    return x, y


@pytest.fixture
def paths(tmp_path) -> dict[str, str]:
    """Checkpoint paths isolated to a temporary directory."""
    return {
        "best_weights_path": str(tmp_path / "best_weights.npy"),
        "last_weights_path": str(tmp_path / "last_weights.npy"),
        "best_value_path": str(tmp_path / "best_value.npy"),
        "value_history_path": str(tmp_path / "value_history.npy"),
    }


@pytest.fixture
def make_trainer(model, paths):
    """Build a compiled trainer, overriding any constructor argument."""

    def _factory(**kwargs) -> InfiniteTrainer:
        options = {**paths, **kwargs}
        trainer = InfiniteTrainer(model=options.pop("model", model), **options)
        trainer.compile(optimizer="adam", loss="mse")
        return trainer

    return _factory
