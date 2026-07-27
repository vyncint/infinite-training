"""Framework-independent training session state and loop.

Internal module: nothing here is part of the public API.

Everything that does not depend on Keras or PyTorch lives here — the target and
timeout bookkeeping, the value history, the checkpoint paths, and the round loop
itself. A backend supplies four small hooks describing how to run one round and
how to move weights in and out of its own model objects.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np

from . import _storage
from .target import Target

logger = logging.getLogger(__name__)

__all__ = ["BaseTrainer"]


class BaseTrainer:
    """Shared state and round loop for every backend.

    Subclasses implement :meth:`_run_round`, :meth:`_current_weights`,
    :meth:`_adopt_best_weights` and :meth:`_best_weights_snapshot`.
    """

    def __init__(
        self,
        target: Target | None,
        timeout: float,
        best_weights_path: str,
        last_weights_path: str,
        best_value_path: str,
        value_history_path: str,
    ) -> None:
        # Target() as a default argument would be shared by every instance.
        self.target = target if target is not None else Target()
        self.timeout = timeout

        self.best_weights_path = best_weights_path
        self.last_weights_path = last_weights_path
        self.best_value_path = best_value_path
        self.value_history_path = value_history_path

        self.best_value = _storage.load_scalar(
            self.best_value_path, self.target.worst_possible_value
        )
        self.value_history = _storage.load_series(self.value_history_path)

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
    # Backend hooks
    # ------------------------------------------------------------------
    def _run_round(self, *args: Any, **kwargs: Any) -> float:
        """Run one training round and return the watched value."""
        raise NotImplementedError

    def _current_weights(self) -> Any:
        """Return the live model's weights in this backend's own format."""
        raise NotImplementedError

    def _adopt_best_weights(self) -> None:
        """Copy the live model's weights into the shadow best-model."""
        raise NotImplementedError

    def _best_weights_snapshot(self) -> Any:
        """Return the shadow best-model's weights."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------
    def _run_session(self, *args: Any, **kwargs: Any) -> None:
        """Run rounds until the target, the timeout, or ``Ctrl+C`` ends it.

        Whichever way it ends, checkpoints are written before returning.
        """
        start = time.time()
        stop_reason = "target reached"
        try:
            while True:
                value = self._run_round(*args, **kwargs)

                self.value_history = np.append(self.value_history, value)

                if self.target.is_improvement(value, self.best_value):
                    self.best_value = value
                    self._adopt_best_weights()

                if self.target.is_reached(value):
                    break
                if (time.time() - start) > self.timeout:
                    stop_reason = "timeout reached"
                    break
        except KeyboardInterrupt:
            stop_reason = "interrupted by user"
            logger.warning("Training interrupted; saving checkpoints before exit.")

        self.last_weights = self._current_weights()
        self.best_weights = self._best_weights_snapshot()
        self.save()
        logger.info(
            "Training stopped (%s) after %d round(s); best %s=%s",
            stop_reason,
            self.rounds_completed,
            self.target.name,
            self.best_value,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self) -> None:
        """Write the best weights, last weights, best value and history to disk."""
        self._save_weights(self.best_weights_path, self.best_weights)
        self._save_weights(self.last_weights_path, self.last_weights)
        _storage.save_object(self.best_value_path, self.best_value)
        _storage.save_object(self.value_history_path, self.value_history)

    def _save_weights(self, path: str, weights: Any) -> None:
        """Persist weights in whatever shape this backend uses."""
        _storage.save_weights(path, weights)


def resolve_timeout(timeout: float | None) -> float:
    """Treat ``None`` as no timeout, so callers can pass it through."""
    return math.inf if timeout is None else timeout
