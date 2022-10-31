from typing import List

from _legacy.data.data_properties import DimensionMixin
from framework.submodel import Block
from submodels.resnet.simple_resnet import SimpleResNet
import tensorflow as tf

from utils.layer_utilities import ResNetV2Block
import numpy as np


class ScalingResNetArtist(SimpleResNet):
    """An Image-to-Image artist, closely following the original ResNetv2 logic."""

    def __init__(self, dimensions: DimensionMixin, filters_each_stage: List,
                 blocks_per_stage: list = None, name="ResNet_Artist", **kwargs):
        self.image_dims = dimensions.get_dimensions()
        super(ScalingResNetArtist, self).__init__(filters_each_stage=filters_each_stage,
                                                  blocks_per_stage=blocks_per_stage,
                                                  number_of_classes=1,
                                                  name=name, **kwargs)

    def _get_factors(self):
        factors = np.ones(len(self.filters[1:]), dtype=int)
        half_half = int(.5 * len(self.filters[1:]))
        factors[:half_half] = 2
        factors[-half_half:] = -2
        factors = factors.tolist()
        return factors

    @staticmethod
    def conv_or_transposed(filters, strides):
        if strides > 0:
            return tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), kernel_initializer='he_normal',
                                          strides=strides)
        elif strides < 0:
            return tf.keras.layers.Conv2DTranspose(filters, kernel_size=(1, 1),
                                                   kernel_initializer='he_normal',
                                                   strides=int(-strides))
        else:
            raise ValueError

    def initialLayers(self):
        return [tf.keras.layers.ZeroPadding2D(padding=(3, 3)),
                tf.keras.layers.Conv2D(self.filters[0], kernel_size=(7, 7), kernel_initializer='he_normal',
                                       strides=2, padding='valid'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
                tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))
                ]

    def innerLayers(self, triple_filter, kernel_size=(3, 3), strides=1):

        return [self.conv_or_transposed(triple_filter[0], strides),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(triple_filter[1], kernel_size=kernel_size,
                                       kernel_initializer='he_normal', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(triple_filter[2], kernel_size=(1, 1), kernel_initializer='he_normal'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU()
                ]

    def convolutionBlock(self, triple_filter, kernel_size, strides, **kwargs):
        if isinstance(triple_filter, int):
            triple_filter = [triple_filter, triple_filter, 4 * triple_filter]  # as in paper
        assert len(triple_filter) == 3
        main_block = Block()
        shortcut_block = Block()
        main_block.extend(self.innerLayers(triple_filter, kernel_size, strides))

        shortcut_block.extend([self.conv_or_transposed(triple_filter[-1], strides),
                               tf.keras.layers.BatchNormalization()])

        return ResNetV2Block(main_block, shortcut_block, name=kwargs['name'] + "Convolution_block")

    def coreLayers(self) -> list:
        core = []
        factors = self._get_factors()
        for stage, (n_filters, stage_size, factor) in enumerate(zip(self.filters[1:], self.blocks_per_stage, factors)):
            core_block = self.coreSegment(n_filters, (3, 3), strides=factor, stage_size=stage_size,
                                          name="stage_{}_".format(stage))
            for block in core_block:
                core.append(block)
        return core

    def finalizingLayers(self):
        return [
            tf.keras.layers.Conv2DTranspose(self.filters[0], kernel_size=(7, 7), kernel_initializer='he_normal',
                                            strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(self.image_dims[2], kernel_size=(7, 7), strides=2, padding='same',
                                            activation=tf.keras.activations.tanh)
        ]


class NonscalingResNetArtist(ScalingResNetArtist):
    def _get_factors(self):
        return [1]*len(self.filters[1:])


class ResNetArtist(ScalingResNetArtist):
    """A Noise-to-Image spin-off of the original ResNet. Also very close to the StyleGAN artist"""

    def __init__(self, dimensions: DimensionMixin, noise_dimension, filters_each_stage: List[int],
                 blocks_per_stage: list = None, name="ResNet_Artist", **kwargs):
        self.noise_dim = noise_dimension
        self.initial_dims = np.ceil(np.sqrt(self.noise_dim))
        super(ResNetArtist, self).__init__(filters_each_stage=filters_each_stage,
                                           blocks_per_stage=blocks_per_stage,
                                           dimensions=dimensions,
                                           name=name, **kwargs)

    def _get_factors(self):
        #  Get closest power of 2 dimension to noise dim
        final_dims = self.image_dims
        factors = [1] * len(
            self.filters[1:])  # this would have been numpy, but gives issues with datatype(int & np.int32)
        number_of_upscales = int(np.log2(final_dims[0] / self.initial_dims))
        for i in range(0, (number_of_upscales + 1)):
            factors[i] = -2
        return factors

    def initialLayers(self):
        noise_reshaping = [tf.keras.layers.Dense(self.initial_dims * self.initial_dims * self.image_dims[-1]),
                           tf.keras.layers.Reshape((self.initial_dims, self.initial_dims, self.image_dims[-1]))]
        return noise_reshaping + super(ResNetArtist, self).initialLayers()
