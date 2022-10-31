from _legacy.data.data_provider import DataProvider
from utils.resize import Resize
import tensorflow as tf


class DataAugmenter(DataProvider):
    """
    TODO: Doesn't work! Get function raises exception: layer has not been called
    """
    def __init__(self, data_provider: DataProvider):
        super(DataAugmenter, self).__init__()
        self.data_provider = data_provider

    def load(self):
        return

    def get(self, n):
        data = self.data_provider.get(n)
        return self.augment(data)

    def get_dimensions(self):
        return self.data_provider.get_dimensions()

    def augment(self, data):
        raise NotImplementedError

    def get_batch(self):
        data = self.data_provider.get_batch()
        return self.augment(data)


class ResizeAugmenter(DataAugmenter):
    def __init__(self, data_provider: DataProvider, size: Resize):
        super(ResizeAugmenter, self).__init__(data_provider=data_provider)
        self.old_shape = self.data_provider.get_dimensions()
        old_shape = self.old_shape[:-1].as_list()
        self.new_shape = size.resize(orig_size=old_shape)
        self.layer = tf.keras.layers.experimental.preprocessing.Resizing(*self.new_shape)

    def augment(self, data):
        return self.layer(data)

    def get_dimensions(self):
        return [*self.new_shape, self.old_shape[-1]]
