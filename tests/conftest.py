"""Shared fixtures for the test suite."""

from __future__ import annotations

import os

# Keep TensorFlow quiet; must be set before the first import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import importlib.util  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402


def _installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


HAS_TENSORFLOW = _installed("tensorflow")
HAS_TORCH = _installed("torch")

# Each backend is optional, so skip the modules whose framework is absent
# rather than failing collection for everyone.
collect_ignore = []
if not HAS_TENSORFLOW:
    collect_ignore += ["test_trainer.py", "test_persistence.py", "test_deprecations.py"]
if not HAS_TORCH:
    collect_ignore += ["test_torch_trainer.py"]


@pytest.fixture
def model():
    """A two-input, one-output functional Keras model small enough to fit instantly."""
    import tensorflow as tf

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
    """Build a compiled Keras trainer, overriding any constructor argument."""
    from infinite_training import InfiniteTrainer

    def _factory(**kwargs) -> InfiniteTrainer:
        options = {**paths, **kwargs}
        trainer = InfiniteTrainer(model=options.pop("model", model), **options)
        trainer.compile(optimizer="adam", loss="mse")
        return trainer

    return _factory


# ----------------------------------------------------------------------
# PyTorch fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def torch_model():
    """A two-input, one-output PyTorch model with deterministic initial weights."""
    import torch
    from torch import nn

    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(2, 1))


@pytest.fixture
def torch_dataset():
    """The same tiny regression problem, as tensors."""
    import torch

    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    return x, y


@pytest.fixture
def make_torch_trainer(torch_model, paths):
    """Build a TorchTrainer, overriding any constructor argument."""
    from infinite_training import TorchTrainer

    def _factory(**kwargs) -> TorchTrainer:
        options = {**paths, **kwargs}
        return TorchTrainer(model=options.pop("model", torch_model), **options)

    return _factory


@pytest.fixture
def make_step(torch_model, torch_dataset):
    """A real gradient-descent step over the tiny dataset."""
    import torch
    from torch import nn

    x, y = torch_dataset

    def _factory(model=None, lr: float = 0.1):
        model = model if model is not None else torch_model
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        def step() -> dict[str, float]:
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            return {"loss": loss.item()}

        return step

    return _factory
