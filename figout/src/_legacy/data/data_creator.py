from abc import ABC
from typing import Tuple, List
import numpy as np
from _legacy.framework.latent import RandomNoiseLatentSpace
from itertools import cycle
import tensorflow as tf


class DataCreator(ABC):
    def create(self, n):
        raise NotImplementedError


class ImageDataCreator(DataCreator, ABC):
    def __init__(self, input_dimensions: Tuple[int, int]):
        self.input_dimensions = input_dimensions


class RandomGrayScaleDataCreator(ImageDataCreator):
    def create(self, n):
        latent = RandomNoiseLatentSpace(self.input_dimensions)
        return latent.get_batch(n)


class SolidGrayScaleDataCreator(ImageDataCreator):
    def get_colors(self, n):
        latent = RandomNoiseLatentSpace(n)
        return latent.get_batch(1)

    def create(self, n):
        colors = self.get_colors(n)
        colors = tf.convert_to_tensor(colors, dtype=tf.float32)
        colors = ((colors / 127.5) - 1)
        ret = np.zeros((n, *self.input_dimensions, 1), np.float32)
        for i in range(n):
            ret[i, :, :, :] = colors[i]
        return tf.convert_to_tensor(ret)


class SingleColorSolidGrayScaleDataCreator(SolidGrayScaleDataCreator):
    def __init__(self, input_dimensions: Tuple[int, int], color: int):
        super(SingleColorSolidGrayScaleDataCreator, self).__init__(input_dimensions)
        self.color = color

    def get_colors(self, n):
        return [self.color for i in range(n)]


class RandomRGBDataCreator(ImageDataCreator):
    def create(self, n):
        latent = RandomNoiseLatentSpace((*self.input_dimensions, 3))
        return latent.get_batch(n)


class SolidRGBDataCreator(ImageDataCreator):
    def get_colors(self, n):
        latent = RandomNoiseLatentSpace((n, 3))
        return latent.get_batch(1)

    def create(self, n):
        colors = self.get_colors(n)
        colors = tf.convert_to_tensor(colors, dtype=tf.float32)
        colors = ((colors / 127.5) - 1)
        ret = np.zeros((n, *self.input_dimensions, 3), np.float32)
        for i in range(n):
            ret[i, :] = colors[i]
        return tf.convert_to_tensor(ret)

class SingleColorSolidRGBDataCreator(SolidRGBDataCreator):
    def __init__(self, input_dimensions: Tuple[int, int], color: Tuple[int, int, int]):
        super(SolidRGBDataCreator, self).__init__(input_dimensions)
        self.color = color

    def get_colors(self, n):
        return [self.color for i in range(n)]


class AlternateColorSolidRGBDataCreator(SolidRGBDataCreator):
    def __init__(self, input_dimensions: Tuple[int, int], colors: List[Tuple[int, int, int]]):
        super(SolidRGBDataCreator, self).__init__(input_dimensions)
        self.colors = colors
        self.iterator = cycle(iter(self.colors))

    def get_colors(self, n):
        return [next(self.iterator) for i in range(n)]
