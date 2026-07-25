"""Resumable, target-driven training loop for Keras models."""

from __future__ import annotations

import logging
import math
import time
import warnings
from typing import Any

import numpy as np
import tensorflow as tf

from . import _storage
from .target import Target

__all__ = ["InfiniteTrainer", "InfinityTraining"]

logger = logging.getLogger(__name__)

_SEPARATOR = "=" * 98

# Old constructor keyword -> new constructor keyword.
_DEPRECATED_KWARGS = {
    "optimize_weight_path": "best_weights_path",
    "last_weight_path": "last_weights_path",
    "optimize_value_path": "best_value_path",
    "list_value_path": "value_history_path",
}


class InfiniteTrainer:
    """Train a Keras model until a target is met, time runs out, or you interrupt.

    The trainer calls :meth:`tf.keras.Model.fit` in a loop. After each round it
    reads the watched quantity from the returned ``History``, records it, and
    keeps a copy of the weights whenever they are the best seen so far. The
    loop ends when the :class:`~infinite_training.Target` is reached, when
    ``timeout`` seconds have elapsed, or when you press ``Ctrl+C``.

    Whichever way it ends, the best weights, the most recent weights, and the
    full value history are written to disk, so a later session picks up where
    the previous one stopped.

    Args:
        model: A compiled-or-not ``tf.keras.Model``. It must be clonable by
            ``tf.keras.models.clone_model`` (functional and sequential models
            are; some subclassed models are not).
        target: Stopping criterion. Defaults to a :class:`Target` that watches
            ``"loss"`` and is never reached, so the session is bounded only by
            ``timeout`` or ``Ctrl+C``.
        timeout: Wall-clock budget in seconds. The check happens *between*
            rounds, so a session can overrun by up to one round.
        best_weights_path: Where the best weights are stored.
        last_weights_path: Where the most recent weights are stored.
        best_value_path: Where the best observed value is stored.
        value_history_path: Where the per-round value history is stored.

    Example:
        >>> model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
        >>> trainer = InfiniteTrainer(
        ...     model=model,
        ...     target=Target("loss", smaller_is_better=True, target_value=1e-4),
        ...     timeout=60,
        ... )
        >>> trainer.compile(optimizer="adam", loss="mse")
        >>> trainer.train(x, y)
        >>> predictions, value = trainer.predict_best(x)

    Note:
        Checkpoints are pickled NumPy files. Only load checkpoints you produced
        yourself; see the security note in the README.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        target: Target | None = None,
        timeout: float = math.inf,
        best_weights_path: str = "optimize_weight.npy",
        last_weights_path: str = "last_weight.npy",
        best_value_path: str = "optimize_value.npy",
        value_history_path: str = "list_value.npy",
        **deprecated: Any,
    ) -> None:
        for old, new in _DEPRECATED_KWARGS.items():
            if old not in deprecated:
                continue
            warnings.warn(
                f"{old!r} is deprecated and will be removed in 3.0.0; use {new!r}.",
                DeprecationWarning,
                stacklevel=2,
            )
            # An explicitly passed legacy path wins over the new-style default.
            value = deprecated.pop(old)
            if old == "optimize_weight_path":
                best_weights_path = value
            elif old == "last_weight_path":
                last_weights_path = value
            elif old == "optimize_value_path":
                best_value_path = value
            else:
                value_history_path = value
        if deprecated:
            unexpected = ", ".join(sorted(deprecated))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

        self.model = model
        # Target() as a default argument would be shared by every instance.
        self.target = target if target is not None else Target()
        self.timeout = timeout

        self.best_weights_path = best_weights_path
        self.last_weights_path = last_weights_path
        self.best_value_path = best_value_path
        self.value_history_path = value_history_path

        self.last_weights = _storage.load_weights(self.last_weights_path, model.get_weights)
        self.best_weights = _storage.load_weights(self.best_weights_path, lambda: self.last_weights)
        self.best_value = _storage.load_scalar(
            self.best_value_path, self.target.worst_possible_value
        )
        self.value_history = _storage.load_series(self.value_history_path)

        self.best_model: tf.keras.Model | None = None

    # ------------------------------------------------------------------
    # Derived state
    # ------------------------------------------------------------------
    @property
    def last_value(self) -> float | None:
        """Value from the most recent round, or ``None`` before any round runs.

        Derived from :attr:`value_history`, so it survives a restart.
        """
        if self.value_history is None or len(self.value_history) == 0:
            return None
        return float(self.value_history[-1])

    @property
    def rounds_completed(self) -> int:
        """How many training rounds have been recorded, across all sessions."""
        return 0 if self.value_history is None else int(len(self.value_history))

    # ------------------------------------------------------------------
    # Keras passthrough
    # ------------------------------------------------------------------
    def compile(self, *args: Any, **kwargs: Any) -> None:
        """Compile the model and prepare the shadow copy holding the best weights.

        Accepts the same arguments as :meth:`tf.keras.Model.compile`. Must be
        called before :meth:`train`.
        """
        self.model.compile(*args, **kwargs)
        self.best_model = tf.keras.models.clone_model(self.model)
        self.best_model.set_weights(self.best_weights)
        self.model.set_weights(self.last_weights)

    def train(self, *args: Any, **kwargs: Any) -> None:
        """Repeatedly call :meth:`tf.keras.Model.fit` until the session ends.

        Accepts the same arguments as ``fit``. Each iteration is one ``fit``
        call, so ``epochs`` (default 1) controls the granularity at which the
        target and timeout are checked.

        Raises:
            RuntimeError: If :meth:`compile` has not been called, or if the
                watched quantity is absent from the training history.
        """
        if self.best_model is None:
            raise RuntimeError("compile() must be called before train().")

        start = time.time()
        stop_reason = "target reached"
        try:
            while True:
                history = self.model.fit(*args, **kwargs)
                value = self._read_target_value(history)

                self.value_history = np.append(self.value_history, value)

                if self.target.is_improvement(value, self.best_value):
                    self.best_value = value
                    self.best_model.set_weights(self.model.get_weights())

                if self.target.is_reached(value):
                    break
                if (time.time() - start) > self.timeout:
                    stop_reason = "timeout reached"
                    break
        except KeyboardInterrupt:
            stop_reason = "interrupted by user"
            logger.warning("Training interrupted; saving checkpoints before exit.")

        self.last_weights = self.model.get_weights()
        self.best_weights = self.best_model.get_weights()
        self.save()
        logger.info(
            "Training stopped (%s) after %d round(s); best %s=%s",
            stop_reason,
            self.rounds_completed,
            self.target.name,
            self.best_value,
        )

    def _read_target_value(self, history: tf.keras.callbacks.History) -> float:
        """Extract the watched quantity from a ``History``, or explain why not."""
        try:
            series = history.history[self.target.name]
        except KeyError:
            available = ", ".join(sorted(history.history)) or "none"
            raise RuntimeError(
                f"Target {self.target.name!r} is not in the training history. "
                f"Available keys: {available}. "
                "Validation keys (val_*) require passing validation_data to train()."
            ) from None
        return float(series[-1])

    def save(self) -> None:
        """Write the best weights, last weights, best value and history to disk."""
        _storage.save_weights(self.best_weights_path, self.best_weights)
        _storage.save_weights(self.last_weights_path, self.last_weights)
        _storage.save_object(self.best_value_path, self.best_value)
        _storage.save_object(self.value_history_path, self.value_history)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_best(self, *args: Any, **kwargs: Any) -> tuple[Any, float]:
        """Predict with the best weights seen.

        Returns:
            The model output and the best observed value.
        """
        self._require_compiled()
        return self.best_model.predict(*args, **kwargs), self.best_value

    def predict_last(self, *args: Any, **kwargs: Any) -> tuple[Any, float | None]:
        """Predict with the most recent weights.

        Returns:
            The model output and the most recent value (``None`` if untrained).
        """
        return self.model.predict(*args, **kwargs), self.last_value

    def show_result(self, *args: Any, **kwargs: Any) -> None:
        """Print predictions from both the best and the most recent weights."""
        self._require_compiled()
        best_output, best_value = self.predict_best(*args, **kwargs)
        last_output, last_value = self.predict_last(*args, **kwargs)
        print()
        print(_SEPARATOR)
        print("Best result: ", best_output)
        print(f"  with {self.target.name}: ", best_value)
        print(_SEPARATOR)
        print("Last result: ", last_output)
        print(f"  with {self.target.name}: ", last_value)
        print(_SEPARATOR)

    def _require_compiled(self) -> None:
        if self.best_model is None:
            raise RuntimeError("compile() must be called before inference.")

    # ------------------------------------------------------------------
    # Deprecated aliases (removed in 3.0.0)
    # ------------------------------------------------------------------
    def predict_optimize(self, *args: Any, **kwargs: Any) -> tuple[Any, float]:
        """Deprecated alias for :meth:`predict_best`."""
        _warn_renamed("predict_optimize()", "predict_best()")
        return self.predict_best(*args, **kwargs)


def _warn_renamed(old: str, new: str) -> None:
    warnings.warn(
        f"{old} is deprecated and will be removed in 3.0.0; use {new}.",
        DeprecationWarning,
        stacklevel=3,
    )


def _deprecated_alias(new_name: str, old_name: str) -> property:
    """Build a property that proxies ``old_name`` to ``new_name`` with a warning."""

    def getter(self: InfiniteTrainer) -> Any:
        _warn_renamed(f"{old_name!r}", f"{new_name!r}")
        return getattr(self, new_name)

    def setter(self: InfiniteTrainer, value: Any) -> None:
        _warn_renamed(f"{old_name!r}", f"{new_name!r}")
        setattr(self, new_name, value)

    getter.__doc__ = f"Deprecated alias for :attr:`{new_name}`."
    return property(getter, setter)


for _old, _new in (
    ("optimize_weight", "best_weights"),
    ("last_weight", "last_weights"),
    ("optimize_value", "best_value"),
    ("list_value", "value_history"),
    ("optimize_model", "best_model"),
    ("optimize_weight_path", "best_weights_path"),
    ("last_weight_path", "last_weights_path"),
    ("optimize_value_path", "best_value_path"),
    ("list_value_path", "value_history_path"),
):
    setattr(InfiniteTrainer, _old, _deprecated_alias(_new, _old))
del _old, _new


class InfinityTraining(InfiniteTrainer):
    """Deprecated name for :class:`InfiniteTrainer`.

    Retained so that existing code keeps working; it will be removed in 3.0.0.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "InfinityTraining is deprecated and will be removed in 3.0.0; use InfiniteTrainer.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
