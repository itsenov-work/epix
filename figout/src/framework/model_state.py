from utils.logger import LoggerMixin
import tensorflow as tf


class ModelState(LoggerMixin):
    runtime_name = ""
    epoch = tf.Variable(0, dtype=tf.int64)
    step = tf.Variable(0, dtype=tf.int64)
    total_steps = tf.Variable(0, dtype=tf.int64)
    # We must call "compile" before starting training
    is_compiled = False

    def __init__(self, model):
        super(ModelState, self).__init__()
        self.model = model
        if "runtime_name" in model.arguments:
            self.runtime_name = model.arguments['runtime_name']

    def increment_step(self):
        self.step.assign_add(1)
        self.total_steps.assign_add(1)

    def increment_epoch(self):
        self.epoch.assign_add(1)
        self.step.assign(0)

    def has_been_compiled(self, *args, **kwargs):
        if self.is_compiled:
            raise Exception("This model has already been compiled!")
        self.is_compiled = True
