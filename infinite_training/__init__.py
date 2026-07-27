"""Train models until a target is met, time runs out, or you interrupt.

``infinite_training`` wraps a training loop in a resumable session that tracks
the best weights it has seen and checkpoints them to disk, so training can be
stopped with ``Ctrl+C`` and continued later.

Two backends are available and they share the same
:class:`~infinite_training.Target`, checkpointing and resume behaviour:

- :class:`~infinite_training.InfiniteTrainer` drives ``tf.keras.Model.fit``.
- :class:`~infinite_training.TorchTrainer` drives a training step you write.

Keras::

    from infinite_training import InfiniteTrainer, Target

    trainer = InfiniteTrainer(
        model=model,
        target=Target("val_accuracy", smaller_is_better=False, target_value=0.98),
        timeout=600,
    )
    trainer.compile(optimizer="adam", loss="mse")
    trainer.train(x, y)
    predictions, best_value = trainer.predict_best(x)

PyTorch::

    from infinite_training import TorchTrainer, Target

    trainer = TorchTrainer(
        model=model,
        target=Target("loss", smaller_is_better=True, target_value=1e-4),
        timeout=600,
    )
    trainer.train(step)          # step() runs one round, returns {"loss": ...}
    predictions, best_value = trainer.predict_best(x)

The backends are imported lazily, so installing only one of TensorFlow and
PyTorch is enough — and importing this package does not pull in a framework
until you actually reference a trainer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .target import Target

__version__ = "2.2.0"

__all__ = [
    "InfiniteTrainer",
    "TorchTrainer",
    "Target",
    # Deprecated, removed in 3.0.0.
    "InfinityTraining",
    "__version__",
]

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from .torch_trainer import TorchTrainer
    from .trainer import InfiniteTrainer, InfinityTraining

# Attribute name -> the submodule that defines it.
_LAZY_ATTRS = {
    "InfiniteTrainer": ".trainer",
    "InfinityTraining": ".trainer",
    "TorchTrainer": ".torch_trainer",
}

# Which framework each submodule needs, for a readable error when it is absent.
_BACKEND_REQUIREMENTS = {
    ".trainer": ("tensorflow", "tensorflow"),
    ".torch_trainer": ("torch", "torch"),
}


def __getattr__(name: str) -> Any:
    """Import a trainer on first use, so only the backend you use is loaded."""
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    try:
        module = import_module(module_name, __name__)
    except ImportError as exc:
        package, extra = _BACKEND_REQUIREMENTS[module_name]
        raise ImportError(
            f"{name} requires {package}, which is not installed. "
            f"Install it with: pip install 'infinite-training[{extra}]'"
        ) from exc

    value = getattr(module, name)
    # Cache on the package so later lookups skip __getattr__ entirely.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
