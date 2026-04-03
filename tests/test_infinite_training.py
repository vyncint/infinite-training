import math
import numpy as np
import tensorflow as tf
from infinite_training import Target, InfinityTraining
import pytest

def test_target_default():
    target = Target()
    assert target.name == "loss"
    assert target.smaller_is_better is True
    assert target.target_value == -math.inf

def test_target_custom():
    target = Target(name="accuracy", smaller_is_better=False, target_value=0.95)
    assert target.name == "accuracy"
    assert target.smaller_is_better is False
    assert target.target_value == 0.95

def test_infinite_training_initialization(tmp_path):
    inputs = tf.keras.Input(shape=(2,))
    x = tf.keras.layers.Dense(4)(inputs)
    outputs = tf.keras.layers.Dense(2)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    target = Target(name="loss", smaller_is_better=True, target_value=0.0001)

    optimize_weight_path = str(tmp_path / "opt.npy")
    last_weight_path = str(tmp_path / "last.npy")
    optimize_value_path = str(tmp_path / "opt_val.npy")
    list_value_path = str(tmp_path / "list_val.npy")

    it = InfinityTraining(
        model=model,
        target=target,
        timeout=10,
        optimize_weight_path=optimize_weight_path,
        last_weight_path=last_weight_path,
        optimize_value_path=optimize_value_path,
        list_value_path=list_value_path
    )

    assert it.timeout == 10
    assert it.optimize_value == math.inf
    assert it.list_value.size == 0

    it.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    assert it.optimize_model is not None

