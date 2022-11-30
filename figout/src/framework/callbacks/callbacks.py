from utils import logging
import os

import tensorflow as tf

from utils.logger import LoggerMixin

logging.getLogger("tensorboard").setLevel(logging.ERROR)
from tensorboard import program


class OurCallback(LoggerMixin, tf.keras.callbacks.Callback):
    def __init__(self):
        super(OurCallback, self).__init__()

    def fetch_extra_inputs(self, **kwargs):
        return


# TODO: Uses legacy FolderDataWriter
#
# class ResultsCallback(OurCallback):
#     def __init__(self, num_results: int, data_writer: FolderDataWriter):
#         super(ResultsCallback, self).__init__()
#         self.num_results = num_results
#         self.data_writer = data_writer
#
#     def on_epoch_end(self, epoch, logs=None):
#         self.log.i(f"Saving results for epoch: {epoch.numpy()}")
#         model_folder = self.model.store.dirs().model_dir
#         images_folder = os.path.join(model_folder, "results_callback")
#         folder = os.path.join(images_folder, "epoch {0:05}".format(epoch.numpy()))
#         os.makedirs(folder, exist_ok=True)
#         model_results = self.model.get_results(self.num_results)
#         for i, results in enumerate(model_results.data):
#             results_folder = os.path.join(folder, str(i))
#             os.makedirs(results_folder, exist_ok=True)
#             self.data_writer.set_folder(results_folder)
#             self.data_writer.write(results, model_results.names)


class CustomLearningRateScheduler(OurCallback):
    """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_end(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))


class GraphModelCallback(OurCallback):
    def on_train_begin(self, logs=None):
        model_folder = self.model.store.dirs().model_dir
        graph_models = self.model.get_functional_models()
        for graph_model in graph_models:
            self.log.i(graph_model.summary())
            tf.keras.utils.plot_model(
                graph_model,
                to_file=os.path.join(model_folder, graph_model.name + ".png"), dpi=96,  # saving
                show_shapes=True, show_layer_names=True,
                expand_nested=True
            )
        del graph_models


class TensorboardCallback(OurCallback):
    subdir = 'metric'

    def __init__(self):
        super(TensorboardCallback, self).__init__()
        self.tb_path = None
        self.summary_writer = None

    def _launch_tensorboard(self):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.tb_path, "--port", "8080"])
        url = tb.launch()

    def on_train_begin(self, logs=None):
        model_folder = self.model.store.dirs().model_dir
        self.tb_path = os.path.join(model_folder, "tensorboard")
        metrics_path = os.path.join(self.tb_path, self.subdir)
        os.makedirs(metrics_path, exist_ok=True)
        self.summary_writer = tf.summary.create_file_writer(metrics_path)
        self._launch_tensorboard()


class GCloudTensorboardCallback(TensorboardCallback):
    def on_train_begin(self, logs=None):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.tb_path])
        url = tb.launch()


class MetricsCallback(TensorboardCallback):
    def __init__(self):
        super(MetricsCallback, self).__init__()
        self.metrics_dict = None

    def on_epoch_end(self, epoch, logs=None):
        with self.summary_writer.as_default():
            for name, metric in self.metrics_dict.items():
                self.log.i("{} value at epoch {}:  {}".format(name, epoch + 1, metric.result()))
                tf.summary.scalar(name, metric.result(), step=epoch)
                metric.reset_states()


class TrainMetricsCallback(MetricsCallback):
    subdir = 'train'

    def on_train_begin(self, logs=None):
        super(TrainMetricsCallback, self).on_train_begin()
        self.metrics_dict = self.model.training_metrics


class ValidationMetricsCallback(MetricsCallback):
    subdir = 'validate'

    def __init__(self):
        super(ValidationMetricsCallback, self).__init__()

    def _launch_tensorboard(self):
        pass

    def on_train_begin(self, logs=None):
        super(ValidationMetricsCallback, self).on_train_begin()
        self.metrics_dict = self.model.validation_metrics

