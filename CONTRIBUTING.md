# Contributing

Thanks for your interest in improving Infinite Training. Bug reports, questions
and pull requests are all welcome.

## Getting set up

TensorFlow wheels lag the newest Python release, so develop on Python 3.10-3.12.

```bash
git clone https://github.com/vyncint/infinite-training.git
cd infinite-training

python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running the checks

The same three checks run in CI, and all of them should pass before you open a
pull request.

```bash
pytest tests -v                    # test suite
ruff check .                       # lint
ruff format --check .              # formatting
```

TensorFlow prints a lot of start-up noise; `TF_CPP_MIN_LOG_LEVEL=3` silences it.
The test suite sets this for you.

To try a change end to end against a real dataset:

```bash
pip install -e ".[example]"
python examples/mnist.py
```

## Making a change

- **Keep the public API stable.** The 2.0.x names are kept as deprecated
  aliases and are covered by `tests/test_deprecations.py`. If you rename
  something, add an alias that emits a `DeprecationWarning`, note it in the
  deprecation table in `CHANGELOG.md`, and leave the removal for the next major
  version.
- **Add a test with every behaviour change.** Bug fixes should come with a test
  that fails before the fix. Group tests by behaviour in a `Test*` class and
  give each test a name that reads as a sentence.
- **Update the docs in the same change.** New or renamed public API belongs in
  the README's API reference; user-visible changes belong in `CHANGELOG.md`
  under *Unreleased*.
- **Explain the "why" in comments.** The code should say what it does; comments
  are for constraints and reasons that are not obvious from reading it.

## Project layout

| Path | Purpose |
| --- | --- |
| `infinite_training/target.py` | `Target` — the stopping criterion and its comparison rules |
| `infinite_training/trainer.py` | `InfiniteTrainer` — the training loop, plus deprecated aliases |
| `infinite_training/_storage.py` | Checkpoint reading and writing (internal) |
| `tests/` | Test suite, one module per concern |
| `examples/` | Runnable examples, not shipped in the wheel |

## Releasing

Maintainers only:

1. Bump `version` in `pyproject.toml` and `__version__` in
   `infinite_training/__init__.py`. They must match.
2. Move the *Unreleased* notes in `CHANGELOG.md` under the new version.
3. Publish a GitHub release tagged `vX.Y.Z`. The publish workflow runs the
   tests, verifies that the tag matches the packaged version, builds and
   uploads to PyPI.

## Code of conduct

Be respectful and constructive. Assume good faith, and keep discussion focused
on the work.
