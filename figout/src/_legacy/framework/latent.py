from abc import ABC
import tensorflow as tf

from _legacy.data.data_provider import DataProvider
from _legacy.data.data_transformer import DataTransformer


class LatentSpace(ABC):
    def get_batch(self, batch_size):
        raise NotImplementedError


class RandomNoiseLatentSpace(LatentSpace):
    def __init__(self, shape):
        self.shape = shape

    def get_batch(self, batch_size):
        return tf.random.normal([batch_size, self.shape])


class DataProviderLatentSpace(LatentSpace):
    def __init__(self, data_provider: DataProvider):
        self.data_provider = data_provider

    def get_batch(self, batch_size):
        return self.data_provider.get(batch_size)


class DataProviderSyncLatentSpace(LatentSpace):
    def __init__(self, data_provider: DataProvider):
        self.last_result = None

        def wrapper(f):
            #TODO: make set_last_result run in separate thread
            def _wrap(batch_size):
                ret = f(batch_size)
                self.set_last_result(ret)
                return ret
            return _wrap

        data_provider.get = wrapper(data_provider.get)

    def set_last_result(self, data):
        self.last_result = data

    def get_batch(self, batch_size):
        return self.last_result


class DataProviderTransformLatentSpace(DataProviderSyncLatentSpace):
    def __init__(self, data_provider: DataProvider, data_transformer: DataTransformer):
        super(DataProviderTransformLatentSpace, self).__init__(data_provider)
        self.transformer = data_transformer

    def set_last_result(self, data):
        data = self.transformer.transform(data)
        super(DataProviderTransformLatentSpace, self).set_last_result(data)
