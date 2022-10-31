import tensorflow as tf
from tensorflow.python.ops.summary_ops_v2 import ResourceSummaryWriter

from utils.easydict import EasyDict


class MetricsDictDeprecated(EasyDict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            self.__setattr__(name, self.create_metric(name))
            return self[name]

    def create_metric(self, name: str) -> tf.keras.metrics.Metric:
        raise NotImplementedError

    def summary(self, writer: ResourceSummaryWriter, epoch: int):
        with writer.as_default():
            for key, value in self.items():
                tf.summary.scalar(key, value.result(), step=epoch)

    def reset(self):
        for value in self.values():
            value.reset_states()


class MetricsDict(EasyDict):

    def create_metric(self, name):
        self.__setattr__(name, self._create_metric(name))

    @staticmethod
    def _create_metric(name: str) -> tf.keras.metrics.Metric:
        raise NotImplementedError

    def summary(self, writer: ResourceSummaryWriter, epoch: int):
        with writer.as_default():
            for key, value in self.items():
                tf.summary.scalar(key, value.result(), step=epoch)

    def reset(self):
        for value in self.values():
            value.reset_states()


class LossMetrics(MetricsDict):
    def __init__(self):
        super(LossMetrics, self).__init__()

    @staticmethod
    def _create_metric(name):
        return tf.keras.metrics.Mean(name, dtype=tf.float32)


class AccuracyMetrics(MetricsDict):
    def __init__(self):
        super(AccuracyMetrics, self).__init__()

    @staticmethod
    def _create_metric(name):
        return tf.keras.metrics.SparseCategoricalAccuracy(name, dtype=tf.float32)
