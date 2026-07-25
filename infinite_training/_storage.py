"""Checkpoint persistence helpers.

Internal module: nothing here is part of the public API.

Checkpoints are stored as ``.npy`` files written with ``allow_pickle=True``,
because Keras weight lists are ragged sequences of arrays that NumPy cannot
represent without object arrays. Loading such a file executes pickle, so a
checkpoint must be treated as trusted input — see the security note in the
README.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")

__all__ = ["load_or_default", "save_object", "save_weights", "load_weights"]


def load_or_default(path: str, default_factory: Callable[[], T]) -> Any:
    """Load ``path`` if it exists, otherwise build a default.

    The default is produced lazily so that callers never pay for building it
    when a checkpoint is present.
    """
    if not os.path.exists(path):
        return default_factory()
    return np.load(path, allow_pickle=True)


def load_weights(path: str, default_factory: Callable[[], list[np.ndarray]]) -> list[np.ndarray]:
    """Load a Keras weight list, falling back to ``default_factory``."""
    if not os.path.exists(path):
        return default_factory()
    loaded = np.load(path, allow_pickle=True)
    # Saved as an object array; Keras expects a plain list of arrays.
    return [np.asarray(w) for w in loaded]


def load_scalar(path: str, default: float) -> float:
    """Load a scalar checkpoint value as a Python float."""
    if not os.path.exists(path):
        return default
    return float(np.load(path, allow_pickle=True))


def load_series(path: str) -> np.ndarray:
    """Load the recorded value history, or an empty array when absent."""
    if not os.path.exists(path):
        return np.array([])
    return np.asarray(np.load(path, allow_pickle=True))


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_object(path: str, value: Any) -> None:
    """Persist an arbitrary Python object as a ``.npy`` file."""
    _ensure_parent_dir(path)
    np.save(path, np.array(value, dtype="object"), allow_pickle=True)


def save_weights(path: str, weights: list[np.ndarray]) -> None:
    """Persist a Keras weight list."""
    save_object(path, weights)
