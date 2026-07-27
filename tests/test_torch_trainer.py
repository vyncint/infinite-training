"""Tests for the PyTorch backend: loop, metrics contract, persistence, inference."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch
from torch import nn

from infinite_training import Target, TorchTrainer


class TestStopConditions:
    def test_stops_once_the_target_is_reached(self, make_torch_trainer, make_step):
        # target_value=inf while minimising is satisfied by any finite loss,
        # so the loop must exit after exactly one round.
        trainer = make_torch_trainer(target=Target("loss", True, math.inf))
        trainer.train(make_step())
        assert trainer.rounds_completed == 1

    def test_stops_on_timeout(self, make_torch_trainer, make_step):
        # A target that can never be reached leaves the timeout as the only exit.
        trainer = make_torch_trainer(target=Target("loss", True, -math.inf), timeout=0)
        trainer.train(make_step())
        # The timeout is checked between rounds, so exactly one round runs.
        assert trainer.rounds_completed == 1

    def test_runs_many_rounds_until_the_target_is_met(self, make_torch_trainer, make_step):
        # A single Linear(2, 1) cannot separate XOR, so its MSE floor is 0.25.
        # 0.26 is reached after roughly twenty rounds; the timeout is only a
        # backstop so a regression here fails instead of hanging.
        trainer = make_torch_trainer(target=Target("loss", True, 0.26), timeout=60)
        trainer.train(make_step(lr=0.5))
        assert trainer.rounds_completed > 1
        assert trainer.best_value < 0.26

    def test_timeout_defaults_to_unbounded(self, make_torch_trainer):
        trainer = make_torch_trainer()
        assert trainer.timeout == math.inf

    def test_keyboard_interrupt_still_checkpoints(self, make_torch_trainer, paths):
        trainer = make_torch_trainer()

        calls = {"n": 0}

        def step() -> dict[str, float]:
            calls["n"] += 1
            if calls["n"] == 3:
                raise KeyboardInterrupt
            return {"loss": 1.0 / calls["n"]}

        trainer.train(step)

        assert trainer.rounds_completed == 2
        assert os.path.exists(paths["best_weights_path"])
        assert os.path.exists(paths["value_history_path"])


class TestBestWeightTracking:
    def test_tracks_the_best_value(self, make_torch_trainer):
        trainer = make_torch_trainer(target=Target("loss", True, 0.05))
        values = iter([0.5, 0.9, 0.2, 0.7, 0.01])
        trainer.train(lambda: {"loss": next(values)})

        assert trainer.best_value == pytest.approx(0.01)
        assert trainer.rounds_completed == 5

    def test_best_weights_are_not_the_last_weights_when_training_worsens(
        self, make_torch_trainer, torch_model
    ):
        trainer = make_torch_trainer(target=Target("loss", True, -math.inf), timeout=0)

        def step() -> dict[str, float]:
            # Move the weights somewhere obvious after the value is recorded.
            with torch.no_grad():
                for param in torch_model.parameters():
                    param.fill_(7.0)
            return {"loss": 0.5}

        trainer.train(step)

        # The single round improved on the initial +inf, so best adopted it.
        best = trainer.best_model.state_dict()
        assert all(torch.all(value == 7.0) for value in best.values())

    def test_maximised_target_keeps_the_largest_value(self, make_torch_trainer):
        trainer = make_torch_trainer(
            target=Target("acc", smaller_is_better=False, target_value=0.99)
        )
        values = iter([0.1, 0.8, 0.4, 0.995])
        trainer.train(lambda: {"acc": next(values)})

        assert trainer.best_value == pytest.approx(0.995)


class TestMetricsContract:
    def test_missing_target_key_explains_what_is_available(self, make_torch_trainer):
        trainer = make_torch_trainer(target=Target("val_loss"))

        with pytest.raises(RuntimeError, match="not in the metrics"):
            trainer.train(lambda: {"loss": 0.5, "acc": 0.9})

    def test_missing_target_key_lists_the_keys(self, make_torch_trainer):
        trainer = make_torch_trainer(target=Target("val_loss"))

        with pytest.raises(RuntimeError, match="acc, loss"):
            trainer.train(lambda: {"loss": 0.5, "acc": 0.9})

    def test_non_mapping_return_is_a_clear_type_error(self, make_torch_trainer):
        trainer = make_torch_trainer()

        with pytest.raises(TypeError, match="must return a mapping"):
            trainer.train(lambda: 0.5)

    def test_sequence_value_uses_the_last_element(self, make_torch_trainer):
        trainer = make_torch_trainer(target=Target("loss", True, 0.15))
        trainer.train(lambda: {"loss": [0.9, 0.5, 0.1]})

        assert trainer.best_value == pytest.approx(0.1)

    def test_empty_sequence_is_rejected(self, make_torch_trainer):
        trainer = make_torch_trainer()

        with pytest.raises(RuntimeError, match="empty sequence"):
            trainer.train(lambda: {"loss": []})

    def test_tensor_value_is_accepted(self, make_torch_trainer):
        trainer = make_torch_trainer(target=Target("loss", True, 0.5))
        trainer.train(lambda: {"loss": torch.tensor(0.25)})

        assert trainer.best_value == pytest.approx(0.25)

    def test_tensor_value_does_not_retain_the_graph(self, make_torch_trainer, torch_model):
        trainer = make_torch_trainer(target=Target("loss", True, math.inf))

        def step():
            # A tensor that requires grad would keep the graph alive if stored.
            loss = (torch_model(torch.tensor([[1.0, 1.0]])) ** 2).sum()
            assert loss.requires_grad
            return {"loss": loss}

        trainer.train(step)
        assert isinstance(trainer.best_value, float)

    def test_numpy_scalar_value_is_accepted(self, make_torch_trainer):
        trainer = make_torch_trainer(target=Target("loss", True, 0.5))
        trainer.train(lambda: {"loss": np.float32(0.25)})

        assert trainer.best_value == pytest.approx(0.25)


class TestDerivedState:
    def test_last_value_is_none_before_training(self, make_torch_trainer):
        assert make_torch_trainer().last_value is None

    def test_rounds_completed_is_zero_before_training(self, make_torch_trainer):
        assert make_torch_trainer().rounds_completed == 0

    def test_last_value_tracks_the_most_recent_round(self, make_torch_trainer):
        trainer = make_torch_trainer(target=Target("loss", True, 0.05))
        values = iter([0.9, 0.4, 0.01])
        trainer.train(lambda: {"loss": next(values)})

        assert trainer.last_value == pytest.approx(0.01)

    def test_step_receives_forwarded_arguments(self, make_torch_trainer):
        trainer = make_torch_trainer(target=Target("loss", True, math.inf))
        seen = {}

        def step(a, *, b):
            seen["args"] = (a, b)
            return {"loss": 0.5}

        trainer.train(step, 1, b=2)
        assert seen["args"] == (1, 2)


class TestPersistence:
    def test_checkpoints_are_written(self, make_torch_trainer, make_step, paths):
        trainer = make_torch_trainer(target=Target("loss", True, math.inf))
        trainer.train(make_step())

        for path in paths.values():
            assert os.path.exists(path), path

    def test_parent_directories_are_created(self, make_torch_trainer, make_step, tmp_path):
        nested = tmp_path / "runs" / "exp1"
        trainer = make_torch_trainer(
            target=Target("loss", True, math.inf),
            best_weights_path=str(nested / "best.npy"),
            last_weights_path=str(nested / "last.npy"),
            best_value_path=str(nested / "value.npy"),
            value_history_path=str(nested / "history.npy"),
        )
        trainer.train(make_step())

        assert (nested / "best.npy").exists()

    def test_session_resumes_from_disk(self, paths, torch_dataset):
        target = Target("loss", True, 0.02)

        torch.manual_seed(0)
        first_model = nn.Sequential(nn.Linear(2, 1))
        first = TorchTrainer(model=first_model, target=target, timeout=0, **paths)
        values = iter([0.9, 0.4])
        first.train(lambda: {"loss": next(values)})

        assert first.rounds_completed == 1

        torch.manual_seed(0)
        second_model = nn.Sequential(nn.Linear(2, 1))
        second = TorchTrainer(model=second_model, target=target, **paths)

        # History, best value and last value all carry over.
        assert second.rounds_completed == 1
        assert second.best_value == pytest.approx(0.9)
        assert second.last_value == pytest.approx(0.9)

    def test_resumed_weights_match_what_was_saved(self, paths, torch_dataset):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(2, 1))
        trainer = TorchTrainer(model=model, target=Target("loss", True, math.inf), **paths)

        def step() -> dict[str, float]:
            with torch.no_grad():
                for param in model.parameters():
                    param.fill_(3.0)
            return {"loss": 0.5}

        trainer.train(step)

        # A fresh model with different initial weights must be overwritten by
        # the checkpoint the constructor loads.
        torch.manual_seed(99)
        resumed_model = nn.Sequential(nn.Linear(2, 1))
        resumed = TorchTrainer(model=resumed_model, **paths)

        assert all(torch.all(v == 3.0) for v in resumed.model.state_dict().values())

    def test_checkpoint_survives_integer_buffers(self, paths):
        """BatchNorm carries an int64 buffer, which must round-trip unchanged."""
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        trainer = TorchTrainer(model=model, target=Target("loss", True, math.inf), **paths)
        trainer.train(lambda: {"loss": 0.5})

        resumed = TorchTrainer(model=nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2)), **paths)
        original = model.state_dict()["1.num_batches_tracked"]
        restored = resumed.model.state_dict()["1.num_batches_tracked"]

        assert restored.dtype == original.dtype
        assert torch.equal(restored, original)


class TestInference:
    def test_predict_best_returns_output_and_value(self, make_torch_trainer, torch_dataset):
        x, _ = torch_dataset
        trainer = make_torch_trainer(target=Target("loss", True, 0.5))
        trainer.train(lambda: {"loss": 0.25})

        output, value = trainer.predict_best(x)

        assert output.shape == (4, 1)
        assert value == pytest.approx(0.25)

    def test_predict_last_returns_none_value_before_training(
        self, make_torch_trainer, torch_dataset
    ):
        x, _ = torch_dataset
        output, value = make_torch_trainer().predict_last(x)

        assert output.shape == (4, 1)
        assert value is None

    def test_inference_does_not_build_a_graph(self, make_torch_trainer, torch_dataset):
        x, _ = torch_dataset
        output, _ = make_torch_trainer().predict_last(x)

        assert not output.requires_grad

    def test_inference_restores_the_previous_training_mode(
        self, make_torch_trainer, torch_dataset, torch_model
    ):
        x, _ = torch_dataset
        trainer = make_torch_trainer()

        torch_model.train()
        trainer.predict_last(x)
        assert torch_model.training

        torch_model.eval()
        trainer.predict_last(x)
        assert not torch_model.training


class TestNoCompileStep:
    def test_best_model_is_ready_without_compile(self, make_torch_trainer):
        # Unlike the Keras trainer there is no compile(); the shadow copy is
        # built in the constructor, so inference works immediately.
        assert make_torch_trainer().best_model is not None

    def test_best_model_is_a_distinct_object(self, make_torch_trainer, torch_model):
        trainer = make_torch_trainer()
        assert trainer.best_model is not torch_model

        with torch.no_grad():
            for param in torch_model.parameters():
                param.fill_(5.0)

        # Mutating the live model must not reach through to the shadow copy.
        assert not any(torch.all(v == 5.0) for v in trainer.best_model.state_dict().values())
