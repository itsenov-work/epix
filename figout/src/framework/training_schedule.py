import time
from typing import List

from utils.logger import LoggerMixin
import tensorflow as tf


class TrainingSchedule(LoggerMixin):

    def __init__(self, model):
        super(TrainingSchedule, self).__init__()
        self.model = model
        self._is_training = False
        self._should_stop = False
        self.callbacks = tf.keras.callbacks.CallbackList(add_progbar=False, model=self.model)

    def add_callbacks(self, callbacks: List[tf.keras.callbacks.Callback]):
        for cb in callbacks:
            cb.model = self.model
            self.callbacks.append(cb)

    def train(self, training_data, validation_data=None, epochs: int = -1, steps: int = 100):
        model = self.model
        if epochs == -1:
            # value for indefinite looping
            self.log.start(f"Training model for infinite epochs and {steps} steps per epoch")
            epochs = 10 ** 15
        else:
            self.log.start(f"Training model for {epochs} epochs and {steps} steps per epoch")

        self._is_training = True
        for callback in self.callbacks:
            callback.fetch_extra_inputs(training_data=training_data, validation_data=validation_data)
        self.callbacks.on_train_begin()
        progbar = tf.keras.utils.Progbar(steps)
        for epoch in range(epochs):
            start = time.time()
            self.callbacks.on_epoch_begin(epoch)
            for i in range(steps):
                self.callbacks.on_batch_begin(i)
                progbar.update(i)
                model.train_step(training_data)
                self.callbacks.on_batch_end(i)
                if self._should_stop:
                    return
            progbar.update(steps, finalize=True)
            if validation_data is not None:
                model.test_step(validation_data)
            self.callbacks.on_epoch_end(epoch)
            model.log.i('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        self.callbacks.on_train_end()
        self.log.end()
