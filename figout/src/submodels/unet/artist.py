from _legacy.data.data_properties import DimensionMixin
from framework.submodel import Block
import tensorflow as tf

from submodels.unet.simple_unet import UNet, _get_kernel_initializer, UpconvBlock, ConvBlock
from utils.layer_utilities import ResNetBlock


class UNetArtist(UNet):
    def __init__(self, data: DimensionMixin, base_filter, downscales: int):
        super(UNetArtist, self).__init__(data.get_dimensions()[-1], base_filter, downscales)

    def coreLayers(self):
        save_the_filters = self.filters.copy()
        core = [ConvBlock(self.filters.pop(0), padding='same'), self.coreSegment()]
        self.filters = save_the_filters
        return core

    def coreSegment(self):
        if len(self.filters) == 0:
            return Block([])
        current_filters = self.filters.pop(0)
        return ResNetBlock(Block([tf.keras.layers.MaxPooling2D((2, 2)),
                                  ConvBlock(current_filters, padding='same'),
                                  self.coreSegment(),
                                  UpconvBlock(current_filters[-1], padding='same'),
                                  ]))

    def finalizingLayers(self) -> list:
        return [tf.keras.layers.Conv2D(filters=self.classes,
                                       kernel_size=(1, 1),
                                       kernel_initializer=_get_kernel_initializer(self.filters[0][0], kernel_size=3),
                                       strides=1,
                                       padding='same',
                                       activation='tanh')]