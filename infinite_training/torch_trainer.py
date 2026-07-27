"""Resumable, target-driven training loop for PyTorch models."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from torch import nn

from . import _storage
from ._base import BaseTrainer, resolve_timeout
from .target import Target

__all__ = ["TorchTrainer"]


def _to_numpy(state: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Convert a state dict to plain arrays, detached from graph and device."""
    return {key: value.detach().cpu().numpy() for key, value in state.items()}


def _to_tensors(state: Mapping[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Convert a NumPy state dict back to tensors."""
    return {key: torch.as_tensor(value) for key, value in state.items()}


class TorchTrainer(BaseTrainer):
    """Train a PyTorch model until a target is met, time runs out, or you interrupt.

    PyTorch has no ``fit``, so you supply the training step yourself: a callable
    that runs one round and returns the metrics for it. The trainer calls it in
    a loop, records the watched quantity, and keeps a copy of the weights
    whenever they are the best seen so far. The loop ends when the
    :class:`~infinite_training.Target` is reached, when ``timeout`` seconds have
    elapsed, or when you press ``Ctrl+C``.

    Whichever way it ends, the best weights, the most recent weights, and the
    full value history are written to disk, so a later session picks up where
    the previous one stopped.

    Unlike :class:`~infinite_training.InfiniteTrainer` there is no ``compile()``
    step — the shadow copy holding the best weights is built in the constructor.

    Args:
        model: Any ``torch.nn.Module``. It is deep-copied once to hold the best
            weights, so it must be copyable.
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
        >>> model = nn.Sequential(nn.Linear(2, 1))
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> loss_fn = nn.MSELoss()
        >>>
        >>> def step():
        ...     model.train()
        ...     optimizer.zero_grad()
        ...     loss = loss_fn(model(x), y)
        ...     loss.backward()
        ...     optimizer.step()
        ...     return {"loss": loss.item()}
        >>>
        >>> trainer = TorchTrainer(
        ...     model=model,
        ...     target=Target("loss", smaller_is_better=True, target_value=1e-4),
        ...     timeout=60,
        ... )
        >>> trainer.train(step)
        >>> predictions, value = trainer.predict_best(x)

    Note:
        Checkpoints are pickled NumPy files, not ``torch.save`` archives. Only
        load checkpoints you produced yourself; see the security note in the
        README.
    """

    def __init__(
        self,
        model: nn.Module,
        target: Target | None = None,
        timeout: float | None = None,
        best_weights_path: str = "best_weights.npy",
        last_weights_path: str = "last_weights.npy",
        best_value_path: str = "best_value.npy",
        value_history_path: str = "value_history.npy",
    ) -> None:
        super().__init__(
            target=target,
            timeout=resolve_timeout(timeout),
            best_weights_path=best_weights_path,
            last_weights_path=last_weights_path,
            best_value_path=best_value_path,
            value_history_path=value_history_path,
        )

        self.model = model

        self.last_weights = _storage.load_state_dict(
            self.last_weights_path, lambda: _to_numpy(model.state_dict())
        )
        self.best_weights = _storage.load_state_dict(
            self.best_weights_path, lambda: self.last_weights
        )

        # No compile() step in PyTorch, so the shadow copy is built here.
        self.best_model = copy.deepcopy(model)
        self.model.load_state_dict(_to_tensors(self.last_weights))
        self.best_model.load_state_dict(_to_tensors(self.best_weights))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, step_fn: Callable[..., Mapping[str, Any]], *args: Any, **kwargs: Any) -> None:
        """Call ``step_fn`` repeatedly until the session ends.

        Each call is one round, so ``step_fn`` decides the granularity at which
        the target and the timeout are checked — typically one epoch, but a
        fixed number of batches works just as well.

        Args:
            step_fn: Runs one round and returns a mapping of metric name to
                value, for example ``{"loss": 0.31}``. A value may also be a
                sequence, in which case the last element is used, matching how
                Keras reports a multi-epoch ``History``.
            *args: Forwarded to ``step_fn`` on every call.
            **kwargs: Forwarded to ``step_fn`` on every call.

        Raises:
            RuntimeError: If the watched quantity is absent from the returned
                metrics.
            TypeError: If ``step_fn`` does not return a mapping.
        """
        self._run_session(step_fn, *args, **kwargs)

    def _run_round(
        self, step_fn: Callable[..., Mapping[str, Any]], *args: Any, **kwargs: Any
    ) -> float:
        metrics = step_fn(*args, **kwargs)
        return self._read_target_value(metrics)

    def _read_target_value(self, metrics: Any) -> float:
        """Extract the watched quantity from a step's metrics, or explain why not."""
        if not isinstance(metrics, Mapping):
            raise TypeError(
                f"The training step must return a mapping of metric name to value, "
                f"for example {{{self.target.name!r}: 0.31}}, but it returned "
                f"{type(metrics).__name__}."
            )
        try:
            value = metrics[self.target.name]
        except KeyError:
            available = ", ".join(sorted(map(str, metrics))) or "none"
            raise RuntimeError(
                f"Target {self.target.name!r} is not in the metrics returned by the "
                f"training step. Available keys: {available}."
            ) from None
        # Accept a per-epoch sequence as well as a scalar, mirroring Keras.
        if isinstance(value, (list, tuple)) or (isinstance(value, np.ndarray) and value.ndim > 0):
            if len(value) == 0:
                raise RuntimeError(
                    f"Target {self.target.name!r} came back as an empty sequence; "
                    "the training step must report at least one value per round."
                )
            value = value[-1]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        return float(value)

    # ------------------------------------------------------------------
    # Backend hooks
    # ------------------------------------------------------------------
    def _current_weights(self) -> dict[str, np.ndarray]:
        return _to_numpy(self.model.state_dict())

    def _adopt_best_weights(self) -> None:
        self.best_model.load_state_dict(self.model.state_dict())

    def _best_weights_snapshot(self) -> dict[str, np.ndarray]:
        return _to_numpy(self.best_model.state_dict())

    def _save_weights(self, path: str, weights: dict[str, np.ndarray]) -> None:
        _storage.save_state_dict(path, weights)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_best(self, *args: Any, **kwargs: Any) -> tuple[Any, float]:
        """Run the best weights in eval mode, without building a graph.

        Returns:
            The model output and the best observed value.
        """
        return self._infer(self.best_model, *args, **kwargs), self.best_value

    def predict_last(self, *args: Any, **kwargs: Any) -> tuple[Any, float | None]:
        """Run the most recent weights in eval mode, without building a graph.

        Returns:
            The model output and the most recent value (``None`` if untrained).
        """
        return self._infer(self.model, *args, **kwargs), self.last_value

    @staticmethod
    def _infer(model: nn.Module, *args: Any, **kwargs: Any) -> Any:
        """Forward pass in eval mode, restoring the previous mode afterwards."""
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                return model(*args, **kwargs)
        finally:
            model.train(was_training)
