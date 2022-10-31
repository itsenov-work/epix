from typing import List

import tensorflow as tf

from _legacy.data.data_properties import DimensionMixin
from framework.submodel import Submodel
from utils.layer_utilities import EqualizeLearningRate as ELR
from tensorflow.keras.layers import Conv2DTranspose, Dense, Reshape, BatchNormalization, LeakyReLU


class Artist(Submodel):
    """A simple generator without a dimension and filter scaling strategy."""
    def __init__(self, filters: List[int], dimensions: DimensionMixin, noise_dimension, name="Vanilla_Artist", **kwargs):
        self.noise_dim = noise_dimension
        self.image_dims = dimensions.get_dimensions()
        super(Artist, self).__init__(filters=filters, name=name, **kwargs)

    def coreSegment(self, filter_number, size_of_kernel):
        return [
            ELR(Conv2DTranspose(filter_number, size_of_kernel, strides=(1, 1), padding='same', use_bias=False)),
            BatchNormalization(),
            LeakyReLU()
        ]

    def initialLayers(self):
        return [
            Dense(self.image_dims[0] * self.image_dims[1] * self.noise_dim),
            Reshape((self.image_dims[0], self.image_dims[1], self.noise_dim))
        ]

    def finalizingLayers(self):
        return [ELR(Conv2DTranspose(self.image_dims[2], 5, padding='same', activation=tf.keras.activations.tanh))]

