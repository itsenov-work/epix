from abc import ABC

from _legacy.images.image_utilities import toGrayscale
import numpy as np


class DataTransformer(ABC):
    def transform(self, data):
        raise NotImplementedError


class GrayScaleTransformer(DataTransformer):
    def transform(self, data):
        data = toGrayscale(data)
        data = np.expand_dims(data, 3)
        return data
