"""
    Apply in example from https://www.tensorflow.org/datasets/keras_example
"""
import tensorflow_datasets as tfds
import tensorflow as tf
from infinite_training import InfinityTraining, Target

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

it = InfinityTraining(model=model, target=Target(
    name="val_sparse_categorical_accuracy", smaller_is_better=False, target_value=0.98), timeout=100)
it.compile(optimizer=tf.keras.optimizers.Adam(0.001),
           loss=tf.keras.losses.SparseCategoricalCrossentropy(
               from_logits=True),
           metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],)
it.train(ds_train, validation_data=ds_test)
it.show_result(ds_train)