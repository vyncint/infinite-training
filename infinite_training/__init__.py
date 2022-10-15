import os
import math
from typing import Any, Tuple
import numpy as np
import tensorflow as tf
import time


class Target():
    """
    `Target` class: is the format of target that we want to stop the training session.

    Attributes:
        name (str): can be (`loss` or any `metric`).
        smaller_is_better (bool): (`True` or `False` that means we want `result of name` (lower or greater).

        target_value (double): stop the training session when: (`smaller_is_better == True` and `result of name < target_value`) or (`smaller_is_better == False` and `result of name > target_value`)
    Example:

    ```python
    target = Target(name = "loss", smaller_is_better = True, target_value = 0.0001)
    ```
    That means: the target is training the model until the loss is lower than 0.0001
    ```python
    target = Target(name = "acc", smaller_is_better = False, target_value = 0.9)
    ```
    That means: the target is training the model until the acc is greater than 0.9
    """

    def __init__(
        self,
        name: str = "loss",
        smaller_is_better: bool = True,
        target_value: np.double = None
    ) -> None:
        self.name = name
        self.smaller_is_better = smaller_is_better
        self.target_value = target_value or (
            -math.inf if self.smaller_is_better else math.inf
        )


class InfinityTraining:
    """
    `InfinityTraining` class: creating training session base on `Target`, timeout or break by `Ctrl + C`.

    Create example training session:
    ```python
    from infinite_training import Target, InfinityTraining
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(2,))
    x = tf.keras.layers.Dense(4)(inputs)
    outputs = tf.keras.layers.Dense(2)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    target = Target(name = "loss", smaller_is_better = True, target_value = 0.0001)
    it = InfinityTraining(model = model, target = target, timeout = 10)
    ```
    Attributes:
        model: tf.keras.Model() is required.
        optimize_weight_path: the path store the optimize model weight `default` "optimize_weight.npy".
        last_weight_path: the path store the last model weight `default` "last_weight.npy".
        optimize_value_path: the path store the optimize value `default` "optimize_value.npy".
        list_value_path: the path store the list value `default` "list_value.npy".
        target: Target object.
        timeout: timeout in seconds.
    Default: After the training session has been ended, four files `optimize_weight.npy`, `last_weight.npy`, `optimize_value.npy` and `list_value.npy`  will be created in the current folder for continuous training.

    We can also modify these file paths by:
    ```python
    it = InfinityTraining(
                            model = model
                            optimize_weight_path = "result/optimize_weight.npy",
                            last_weight_path = "result/last_weight.npy",
                            optimize_value_path = "result/optimize_value.npy",
                            list_value_path = "result/list_value.npy"
                        )
    ```
    After creating training session we need to `compile`, `train` and `save`.
    ```python
    it.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    it.train(x = x, y = y, batch_size = batch_size, verbose = verbose)
    it.save()
    ```
    Finally, we can `predict_optimize`, `predict_last` or `show_result` as:
    ```python
    it.show_result(x = x)
    optimize_result, optimize_value = it.predict_optimize(x = x)
    last_result, last_value = it.predict_last(x = x)
    ```
    """

    def __load(
            self,
            path: str,
            default: Any
    ) -> Any:
        if os.path.exists(path):
            return np.load(path, allow_pickle=True)
        return default

    def __save(self) -> None:
        np.save(self.optimize_weight_path, np.array(
            self.optimize_weight, dtype="object"), allow_pickle=True)
        np.save(self.last_weight_path, np.array(
            self.last_weight, dtype="object"), allow_pickle=True)
        np.save(self.optimize_value_path, np.array(
            self.optimize_value, dtype="object"), allow_pickle=True)
        np.save(self.list_value_path, np.array(
            self.list_value, dtype="object"), allow_pickle=True)

    def __init__(
        self,
        model: tf.keras.Model(),
        optimize_weight_path: str = "optimize_weight.npy",
        last_weight_path: str = "last_weight.npy",
        optimize_value_path: str = "optimize_value.npy",
        list_value_path: str = "list_value.npy",
        target: Target = Target(),
        timeout: np.double = math.inf
    ) -> None:
        self.model = model
        self.optimize_weight_path = optimize_weight_path
        self.last_weight_path = last_weight_path
        self.optimize_value_path = optimize_value_path
        self.list_value_path = list_value_path
        self.target = target
        self.timeout = timeout
        self.last_weight = self.__load(
            path=self.last_weight_path,
            default=self.model.get_weights()
        )
        self.optimize_weight = self.__load(
            path=self.optimize_weight_path,
            default=self.last_weight
        )
        self.optimize_value = self.__load(
            path=self.optimize_value_path,
            default=math.inf if self.target.smaller_is_better else -math.inf
        )
        self.list_value = self.__load(
            path=self.list_value_path,
            default=np.array([])
        )

    def compile(
        self,
        *args,
        **kwargs
    ) -> None:
        """
        Similar to `tensorflow.keras.Model.compile()` https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
        """
        self.model.compile(*args, **kwargs)
        self.optimize_model = tf.keras.models.clone_model(self.model)
        self.optimize_model.set_weights(self.optimize_weight)
        self.model.set_weights(self.last_weight)

    def train(
            self,
            *args,
            **kwargs
    ) -> None:
        """
        Similar to `tensorflow.keras.Model.fit()` https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        """
        start = time.time()
        try:
            while True:
                history = self.model.fit(*args, **kwargs)
                current_value = history.history[self.target.name]
                self.last_value = current_value[-1]
                self.list_value = np.append(self.list_value, self.last_value)

                if self.target.smaller_is_better:
                    ok = self.optimize_value > self.last_value
                    is_break = self.last_value < self.target.target_value
                else:
                    ok = self.optimize_value < self.last_value
                    is_break = self.last_value > self.target.target_value

                if ok:
                    self.optimize_value = self.last_value
                    self.optimize_model.set_weights(self.model.get_weights())

                if is_break:
                    break
                if (time.time() - start) > self.timeout:
                    break

        except KeyboardInterrupt:
            pass
        self.last_weight = self.model.get_weights()
        self.optimize_weight = self.optimize_model.get_weights()
        self.__save()

    def show_result(
        self,
        *args,
        **kwargs
    ) -> None:
        """
        Similar to `tensorflow.keras.Model.predict()` https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
        """
        print()
        print("==================================================================================================")
        print("Optimize result: ", self.optimize_model.predict(*args, **kwargs))
        print(f'  with {self.target.name}: ', self.optimize_value)
        print("==================================================================================================")
        print("Last result: ", self.model.predict(*args, **kwargs))
        print(f'  with {self.target.name}: ', self.last_value)
        print("==================================================================================================")

    def predict_optimize(
        self,
        *args,
        **kwargs
    ) -> tuple():
        """
        Similar to `tensorflow.keras.Model.predict()` https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
        """
        return (self.optimize_model.predict(*args, **kwargs), self.optimize_value)

    def predict_last(
        self,
        *args,
        **kwargs
    ) -> tuple():
        """
        Similar to `tensorflow.keras.Model.predict()` https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
        """
        return (self.model.predict(*args, **kwargs), self.last_value)