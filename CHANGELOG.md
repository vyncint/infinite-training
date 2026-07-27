# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2026-07-28

Adds a PyTorch backend. Fully backward compatible: the Keras API is unchanged
and no existing checkpoint needs converting.

### Added

- `TorchTrainer`, a PyTorch backend sharing `Target`, the round loop, the
  timeout, the `Ctrl+C` handling and the resume behaviour with the Keras
  trainer. PyTorch has no `fit`, so you pass a step function that runs one round
  and returns its metrics.
- A `torch` extra: `pip install "infinite-training[torch]"`.
- A `tensorflow` extra, so new code can already name the dependency it wants.
  TensorFlow remains a hard requirement for the whole 2.x line; it moves to the
  extra in 3.0.0.
- `examples/mnist_torch.py`, the PyTorch counterpart of the Keras MNIST example.
- Tests for the PyTorch backend and for backend isolation (39 new, 94 in total),
  plus a CI job that installs each framework on its own and runs the suite.

### Changed

- Backends are imported lazily. `import infinite_training` no longer imports
  TensorFlow, and referencing `TorchTrainer` never imports it either — so having
  only one of the two frameworks installed is enough. Asking for a trainer whose
  framework is missing raises an `ImportError` naming the extra to install.
- The loop, the target and timeout bookkeeping, the value history and the
  checkpoint paths moved into an internal `_base` module shared by both
  backends. No public name changed.

## [2.1.0] - 2026-07-25

Backward compatible with 2.0.0: existing code keeps working and warns where an
API has been renamed.

### Fixed

- `Target(target_value=0)` is now honoured. The default was selected with
  `target_value or <bound>`, so a falsy `0`/`0.0` was replaced by an unreachable
  bound and the session could never stop on that target.
- `last_value` no longer raises `AttributeError` when read before training, or
  after an interrupt during the first round. It is derived from the persisted
  value history, so it also survives a restart.
- `Target()` is no longer used as a default argument value, which meant every
  trainer built without an explicit target shared one mutable instance.
- Calling `train()` or an inference method before `compile()` now raises a clear
  `RuntimeError` instead of `AttributeError: 'InfiniteTrainer' object has no
  attribute 'best_model'`.
- A target naming a metric that is absent from the training history now raises a
  `RuntimeError` listing the available keys, instead of a bare `KeyError`. When
  the missing key starts with `val_`, the message points at `validation_data`.
- Checkpoint parent directories are created automatically; previously a path
  such as `runs/exp1/best.npy` failed unless the directory already existed.
- Restored checkpoint values are converted back to Python floats and plain
  weight lists rather than being left as zero-dimensional object arrays.
- `KeyboardInterrupt` is logged instead of silently swallowed.
- `requires-python` corrected from `>=3.9` to `>=3.10`. TensorFlow has required
  3.10 or newer since 2.16, so the old floor advertised an uninstallable
  combination.

### Added

- `InfiniteTrainer`, the new name for `InfinityTraining`.
- `rounds_completed` property.
- `save()` as a public method for checkpointing on demand.
- `Target.is_improvement()`, `Target.is_reached()` and
  `Target.worst_possible_value`, so the comparison rules live with the target.
- `__version__` and `__all__` exports.
- A CI workflow running lint, tests on Python 3.10-3.12, and a packaging check
  on every push and pull request. Previously tests ran only when a release was
  published.
- The publish workflow verifies that the release tag matches the packaged
  version.
- `CONTRIBUTING.md`, `CHANGELOG.md`, and a rewritten `README.md` with an API
  reference, a resume guide and a security note.
- Test suite covering targets, stop conditions, persistence and resume,
  error handling, and the deprecation shims (55 tests, up from 3).

### Changed

- The package is split into `target.py`, `trainer.py` and `_storage.py`. Public
  imports are unchanged: `from infinite_training import InfiniteTrainer, Target`.
- Type hints use `float` rather than `np.double` for scalar values.
- Progress and stop reasons are reported through the `logging` module.
- `run.py` moved to `examples/mnist.py` and no longer ships inside the wheel.
- Packaging declares its packages explicitly, so `tests` is no longer picked up
  by auto-discovery.

### Removed

- The unused `polars` runtime dependency.

### Deprecated

The following still work and will be removed in 3.0.0:

| Deprecated | Replacement |
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

## [2.0.0] - 2026-04-03

- Upgraded to the latest TensorFlow and switched the release workflow to `uv`.

[2.2.0]: https://github.com/vyncint/infinite-training/releases/tag/v2.2.0
[2.1.0]: https://github.com/vyncint/infinite-training/releases/tag/v2.1.0
[2.0.0]: https://github.com/vyncint/infinite-training/releases/tag/v2.0.0
