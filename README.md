# Infinite Training

[![CI](https://github.com/vyncint/infinite-training/actions/workflows/ci.yml/badge.svg)](https://github.com/vyncint/infinite-training/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/infinite-training.svg)](https://pypi.org/project/infinite-training/)
[![Python versions](https://img.shields.io/pypi/pyversions/infinite-training.svg)](https://pypi.org/project/infinite-training/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Train a Keras model until it is *good enough*, until you run out of time, or until you press `Ctrl+C` — then pick up exactly where you left off.

`Model.fit` makes you choose the number of epochs up front. Often what you actually want is "keep going until validation accuracy passes 0.98", or "train for ten minutes and keep the best result". `infinite_training` wraps `fit` in a resumable loop that does that, remembers the best weights it has seen, and checkpoints everything to disk so an interrupted run is never wasted.

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Resuming a session](#resuming-a-session)
- [API reference](#api-reference)
- [Choosing a target](#choosing-a-target)
- [Security note on checkpoints](#security-note-on-checkpoints)
- [Migrating from 2.0.x](#migrating-from-20x)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

```bash
pip install infinite-training
```

To run the bundled example as well:

```bash
pip install "infinite-training[example]"
```

Requires Python 3.10+ and TensorFlow.

---

## Quick start

```python
import numpy as np
import tensorflow as tf
from infinite_training import InfiniteTrainer, Target

x = np.random.rand(256, 2)
y = x.sum(axis=1, keepdims=True)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)

trainer = InfiniteTrainer(
    model=model,
    target=Target(name="loss", smaller_is_better=True, target_value=1e-4),
    timeout=60,  # seconds
)

trainer.compile(optimizer="adam", loss="mse")
trainer.train(x, y, epochs=5, verbose=0)  # loops until the target, the timeout, or Ctrl+C

predictions, best_loss = trainer.predict_best(x, verbose=0)
print(f"best loss {best_loss:.6f} after {trainer.rounds_completed} round(s)")
```

`compile` and `train` forward their arguments to `Model.compile` and `Model.fit`, so anything you already pass to Keras keeps working.

---

## How it works

Each iteration of the loop is one call to `Model.fit`:

1. **Fit** — call `Model.fit(*args, **kwargs)` once. With the default `epochs=1`, one round is one epoch; pass `epochs=5` to check the target less often and reduce overhead.
2. **Read** — take the final value of `target.name` from the returned `History` and append it to `value_history`.
3. **Remember** — if it beats the best value so far, copy the current weights into the best-weights model.
4. **Decide** — stop if the target is reached, or if `timeout` seconds have elapsed.
5. **Checkpoint** — when the loop ends, for any reason, write best weights, last weights, best value and history to disk.

`Ctrl+C` is caught and treated as a normal stop, so the checkpoint still gets written.

> **Timeout granularity.** The elapsed-time check happens *between* rounds, never inside `fit`. A session can therefore overrun `timeout` by up to the duration of one round. Use a smaller `epochs` for a tighter bound.

---

## Resuming a session

Checkpoints are written to four files in the working directory by default. Point them anywhere you like:

```python
trainer = InfiniteTrainer(
    model=model,
    best_weights_path="runs/exp1/best_weights.npy",
    last_weights_path="runs/exp1/last_weights.npy",
    best_value_path="runs/exp1/best_value.npy",
    value_history_path="runs/exp1/value_history.npy",
)
```

Parent directories are created automatically. Construct a trainer with the same paths and it reloads the previous state:

```python
trainer = InfiniteTrainer(model=build_model(), best_weights_path="runs/exp1/best_weights.npy", ...)
trainer.rounds_completed   # e.g. 12 — carried over from the previous session
trainer.last_value         # available immediately, no re-training needed
trainer.compile(optimizer="adam", loss="mse")   # restores both weight sets
trainer.train(x, y)        # continues, appending to the same history
```

`compile()` loads the last weights into `model` and the best weights into `best_model`, so training continues from where it stopped while the best result stays intact.

---

## API reference

### `Target`

The stopping criterion.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | `"loss"` | Key to read from the Keras `History`. Any loss or metric, including `val_*` keys. |
| `smaller_is_better` | `bool` | `True` | `True` for losses and error rates, `False` for accuracy-like metrics. |
| `target_value` | `float \| None` | `None` | Value at which training stops. `None` means an unreachable bound, so the session is limited only by `timeout` or `Ctrl+C`. |

Methods: `is_improvement(candidate, incumbent)`, `is_reached(value)`, and the `worst_possible_value` property.

### `InfiniteTrainer`

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | `tf.keras.Model` | required | Must be clonable by `tf.keras.models.clone_model`. |
| `target` | `Target` | `Target()` | Stopping criterion. |
| `timeout` | `float` | `math.inf` | Wall-clock budget in seconds, checked between rounds. |
| `best_weights_path` | `str` | `"optimize_weight.npy"` | Best weights checkpoint. |
| `last_weights_path` | `str` | `"last_weight.npy"` | Most recent weights checkpoint. |
| `best_value_path` | `str` | `"optimize_value.npy"` | Best observed value. |
| `value_history_path` | `str` | `"list_value.npy"` | Per-round value history. |

| Method | Description |
| --- | --- |
| `compile(*args, **kwargs)` | Forwards to `Model.compile`, then restores both weight sets. Call before `train`. |
| `train(*args, **kwargs)` | Forwards to `Model.fit`, looping until the target, the timeout, or `Ctrl+C`. Always checkpoints. |
| `save()` | Write all four checkpoints immediately. |
| `predict_best(*args, **kwargs)` | `(predictions, best_value)` using the best weights. |
| `predict_last(*args, **kwargs)` | `(predictions, last_value)` using the most recent weights. |
| `show_result(*args, **kwargs)` | Print both sets of predictions side by side. |

| Property | Description |
| --- | --- |
| `best_value` | Best value observed, across all sessions. |
| `last_value` | Value from the most recent round, or `None` before any round has run. |
| `value_history` | NumPy array with one entry per round. |
| `rounds_completed` | Number of recorded rounds. |
| `best_weights` / `last_weights` | Weight lists. |
| `best_model` | Shadow model holding the best weights (available after `compile`). |

---

## Choosing a target

```python
Target("loss", smaller_is_better=True, target_value=0.01)  # stop below 0.01 loss
Target("val_accuracy", smaller_is_better=False, target_value=0.98)  # stop above 98%
Target("loss", smaller_is_better=True, target_value=0.0)  # stop at a negative loss
Target()  # no target: timeout or Ctrl+C only
```

A `val_*` target requires passing `validation_data` to `train`, otherwise the key is absent from the history and the trainer raises a `RuntimeError` listing the keys that *are* available.

---

## Security note on checkpoints

Checkpoints are `.npy` files written with `allow_pickle=True`, which is required because Keras weight lists are ragged. **Loading a checkpoint executes pickle**, so only load checkpoint files you produced yourself. Never point a trainer at checkpoint paths supplied by an untrusted party.

---

## Migrating from 2.0.x

Version 2.1.0 is backward compatible: the 2.0.x API still works and emits `DeprecationWarning`. The old names will be removed in 3.0.0.

| 2.0.x | 2.1.0 |
| --- | --- |
| `InfinityTraining` | `InfiniteTrainer` |
| `optimize_weight` | `best_weights` |
| `last_weight` | `last_weights` |
| `optimize_value` | `best_value` |
| `list_value` | `value_history` |
| `optimize_model` | `best_model` |
| `predict_optimize()` | `predict_best()` |
| `optimize_weight_path=` | `best_weights_path=` |
| `last_weight_path=` | `last_weights_path=` |
| `optimize_value_path=` | `best_value_path=` |
| `list_value_path=` | `value_history_path=` |

Two behaviour fixes may affect you, both listed in the [CHANGELOG](CHANGELOG.md):

- `Target(target_value=0)` is now honoured. Previously `0` was treated as "unset", turning the target into an unreachable bound.
- `last_value` returns `None` before the first round instead of raising `AttributeError`.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, tests and coding standards. Bug reports and pull requests are welcome on the [issue tracker](https://github.com/vyncint/infinite-training/issues).

---

## License

Released under the [MIT License](LICENSE).
