"""Train Keras models until a target is met, time runs out, or you interrupt.

``infinite_training`` wraps ``Model.fit`` in a resumable loop that tracks the
best weights it has seen and checkpoints them to disk, so training can be
stopped with ``Ctrl+C`` and continued in a later session.

Typical use::

    from infinite_training import InfiniteTrainer, Target

    trainer = InfiniteTrainer(
        model=model,
        target=Target("val_accuracy", smaller_is_better=False, target_value=0.98),
        timeout=600,
    )
    trainer.compile(optimizer="adam", loss="mse")
    trainer.train(x, y)
    predictions, best_value = trainer.predict_best(x)
"""

from __future__ import annotations

from .target import Target
from .trainer import InfiniteTrainer, InfinityTraining

__version__ = "2.1.0"

__all__ = [
    "InfiniteTrainer",
    "Target",
    # Deprecated, removed in 3.0.0.
    "InfinityTraining",
    "__version__",
]
