"""Stopping criteria for an infinite training session."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Target"]


@dataclass
class Target:
    """A stopping criterion evaluated after every training round.

    A target describes *which* quantity to watch, in which *direction* it
    improves, and the *value* at which training may stop.

    Args:
        name: Key to read from the Keras ``History``. Any loss or metric name
            produced by ``Model.fit`` is valid, including validation keys such
            as ``"val_loss"`` or ``"val_sparse_categorical_accuracy"``.
        smaller_is_better: ``True`` for quantities minimised (losses, error
            rates), ``False`` for quantities maximised (accuracy, F1).
        target_value: Value at which training stops. When omitted, it defaults
            to an unreachable bound (``-inf`` when minimising, ``+inf`` when
            maximising), so the session runs until it times out or is
            interrupted.

    Examples:
        Train until the loss drops below 0.0001::

            Target(name="loss", smaller_is_better=True, target_value=0.0001)

        Train until accuracy exceeds 0.9::

            Target(name="acc", smaller_is_better=False, target_value=0.9)

        Train indefinitely, stopping only on timeout or ``Ctrl+C``::

            Target()
    """

    name: str = "loss"
    smaller_is_better: bool = True
    target_value: float | None = None

    def __post_init__(self) -> None:
        # An explicit 0.0 (or 0) is a legitimate target, so test for None
        # rather than truthiness.
        if self.target_value is None:
            self.target_value = -math.inf if self.smaller_is_better else math.inf

    @property
    def worst_possible_value(self) -> float:
        """The value an as-yet-unmeasured best score starts from."""
        return math.inf if self.smaller_is_better else -math.inf

    def is_improvement(self, candidate: float, incumbent: float) -> bool:
        """Return ``True`` when ``candidate`` beats ``incumbent``."""
        if self.smaller_is_better:
            return candidate < incumbent
        return candidate > incumbent

    def is_reached(self, value: float) -> bool:
        """Return ``True`` when ``value`` satisfies the stopping criterion."""
        if self.smaller_is_better:
            return value < self.target_value
        return value > self.target_value
