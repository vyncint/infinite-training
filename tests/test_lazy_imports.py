"""The two backends must stay independent of each other.

Installing only TensorFlow, or only PyTorch, has to be enough. These tests run
each import in a subprocess so that modules already loaded by the rest of the
suite cannot mask a regression.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest

HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None

BLOCK_MODULE = """
import sys

class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == {blocked!r} or name.startswith({blocked!r} + "."):
            raise ImportError("simulated: {blocked} is not installed")
        return None

sys.meta_path.insert(0, _Blocker())
"""


def run_snippet(*parts: str) -> subprocess.CompletedProcess[str]:
    """Execute the concatenated parts in a clean interpreter.

    Each part is dedented on its own, so a flush-left prefix can be combined
    with an indented literal without confusing ``textwrap.dedent``.
    """
    code = "".join(textwrap.dedent(part) for part in parts)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"},
    )


def test_importing_the_package_loads_no_framework():
    result = run_snippet(
        """
        import sys
        import infinite_training
        assert infinite_training.Target is not None
        print("tensorflow" in sys.modules, "torch" in sys.modules)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False False"


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_trainer_does_not_import_tensorflow():
    result = run_snippet(
        """
        import sys
        from infinite_training import TorchTrainer
        assert TorchTrainer is not None
        print("tensorflow" in sys.modules)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="requires tensorflow")
def test_keras_trainer_does_not_import_torch():
    result = run_snippet(
        """
        import sys
        from infinite_training import InfiniteTrainer
        assert InfiniteTrainer is not None
        print("torch" in sys.modules)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_missing_torch_gives_an_actionable_error():
    result = run_snippet(
        BLOCK_MODULE.format(blocked="torch"),
        """
        try:
            from infinite_training import TorchTrainer
        except ImportError as exc:
            print(exc)
        else:
            raise AssertionError("expected ImportError")
        """,
    )
    assert result.returncode == 0, result.stderr
    assert "requires torch" in result.stdout
    assert "infinite-training[torch]" in result.stdout


def test_missing_tensorflow_gives_an_actionable_error():
    result = run_snippet(
        BLOCK_MODULE.format(blocked="tensorflow"),
        """
        try:
            from infinite_training import InfiniteTrainer
        except ImportError as exc:
            print(exc)
        else:
            raise AssertionError("expected ImportError")
        """,
    )
    assert result.returncode == 0, result.stderr
    assert "requires tensorflow" in result.stdout
    assert "infinite-training[tensorflow]" in result.stdout


def test_unknown_attribute_still_raises_attribute_error():
    import infinite_training

    with pytest.raises(AttributeError, match="no attribute 'NotATrainer'"):
        getattr(infinite_training, "NotATrainer")  # noqa: B009


def test_dir_lists_the_public_api():
    import infinite_training

    assert "TorchTrainer" in dir(infinite_training)
    assert "InfiniteTrainer" in dir(infinite_training)
