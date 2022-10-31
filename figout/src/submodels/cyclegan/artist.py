from tensorflow_addons.layers import InstanceNormalization

from _legacy.data.data_properties import DimensionMixin
from framework.submodel import Block, Submodel
import tensorflow as tf

from utils.layer_utilities import ReflectionPadding2D, ResNetBlock


class CycleArtist(Submodel):
    """Image-to-image Artist, true to the original CycleGAN architecture, with customizable complexity.
    Params:
        initial_filter: int;    the first filter; the memory usage will not exceed HxWx(initial_filter) throughout.
        downsamples: int;       the number of times the image dimensions are halved, and the filter number is
                                correspondingly doubled.
        resnet_blocks: int;     the core part of the architecture, which doesn't affect dimensions, so increasing
                                this number will only add to the architecture complexity.
        dimensions: data;       takes the data object and extracts the image channels from that.
    """
    def __init__(self, initial_filter: int, downsamples: int, resnet_blocks: int, dimensions: DimensionMixin,
                 name="Cycle_Artist", **kwargs):
        self.downsample_filters = [initial_filter * (2 ** i) for i in range(downsamples)]
        filters = [self.downsample_filters[-1]] * resnet_blocks
        self.output_channels = dimensions.get_dimensions()[2]
        super(CycleArtist, self).__init__(filters, name=name, **kwargs)

    def initialLayers(self):
        initial = [tf.keras.layers.Conv2D(self.downsample_filters[0], (7, 7), padding='same'),
                   InstanceNormalization(axis=-1),
                   tf.keras.layers.ReLU()]
        for filter_v in self.downsample_filters:
            initial.append(tf.keras.layers.Conv2D(filter_v, (3, 3), strides=2, padding='same', use_bias=False))
            initial.append(tf.keras.layers.ReLU())

        return initial

    def coreSegment(self, filter_number, size_of_kernel):
        return [ReflectionPadding2D(padding=(1, 1)),
                tf.keras.layers.Conv2D(filter_number, size_of_kernel, padding='valid', use_bias=False),
                InstanceNormalization(axis=-1),
                tf.keras.layers.ReLU(),
                ReflectionPadding2D(padding=(1, 1)),
                tf.keras.layers.Conv2D(filter_number, size_of_kernel, padding='valid', use_bias=False),
                InstanceNormalization(axis=-1)
                ]

    def coreLayers(self):
        core = []
        for n_filters in self.filters:
            core_block = ResNetBlock(Block(self.coreSegment(n_filters, 3)))
            core.append(core_block)
        return core

    def finalizingLayers(self):
        final = []
        for filter in self.downsample_filters[::-1]:
            final.append(tf.keras.layers.Conv2DTranspose(filter, (3, 3), strides=2, padding='same', use_bias=False))
            final.append(tf.keras.layers.ReLU())
        final.append(ReflectionPadding2D(padding=(3, 3)))
        final.append(tf.keras.layers.Conv2D(self.output_channels, (7, 7), padding='valid', use_bias=False,
                                            activation='tanh'))
        return final
