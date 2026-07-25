"""Train an MNIST classifier until it reaches 98% validation accuracy.

Adapted from https://www.tensorflow.org/datasets/keras_example.

Run it with::

    pip install "infinite_training[example]"
    python examples/mnist.py

Stop it at any point with ``Ctrl+C``: the best and most recent weights are
written to ``models/`` and picked up automatically the next time you run it.
"""

from __future__ import annotations

import tensorflow as tf
import tensorflow_datasets as tfds

from infinite_training import InfiniteTrainer, Target

CHECKPOINT_DIR = "models"
BATCH_SIZE = 128


def normalize_img(image, label):
    """Normalize images: `uint8` -> `float32` in the range [0, 1]."""
    return tf.cast(image, tf.float32) / 255.0, label


def build_datasets():
    """Load MNIST and apply the standard input pipeline."""
    (ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.batch(BATCH_SIZE)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(BATCH_SIZE)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test


def build_model() -> tf.keras.Model:
    """A small dense classifier for 28x28 grayscale digits."""
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )


def main() -> None:
    ds_train, ds_test = build_datasets()

    trainer = InfiniteTrainer(
        model=build_model(),
        target=Target(
            name="val_sparse_categorical_accuracy",
            smaller_is_better=False,
            target_value=0.98,
        ),
        timeout=100,
        best_weights_path=f"{CHECKPOINT_DIR}/best_weights.npy",
        last_weights_path=f"{CHECKPOINT_DIR}/last_weights.npy",
        best_value_path=f"{CHECKPOINT_DIR}/best_value.npy",
        value_history_path=f"{CHECKPOINT_DIR}/value_history.npy",
    )
    trainer.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    trainer.train(ds_train, validation_data=ds_test)
    trainer.show_result(ds_train)


if __name__ == "__main__":
    main()
