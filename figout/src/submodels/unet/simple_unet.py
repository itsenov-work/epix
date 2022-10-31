from typing import List

from tensorflow.python.keras.initializers.initializers_v2 import TruncatedNormal

from framework.submodel import Submodel, Block
import tensorflow as tf
import numpy as np

from utils.layer_utilities import ResNetBlock


def _get_kernel_initializer(filters, kernel_size):
    stddev = np.sqrt(2 / (kernel_size ** 2 * filters))
    return TruncatedNormal(stddev=stddev)


class UpconvBlock(tf.keras.layers.Layer):

    def __init__(self, filters, upscale: int = 2, kernel_size=None, padding='valid', activation="relu",
                 **kwargs):
        super(UpconvBlock, self).__init__(**kwargs)
        self.filters = filters
        if kernel_size is None:
            self.kernel_size = upscale
        else:
            self.kernel_size = kernel_size
        self.pool_size = upscale
        self.padding = padding
        self.activation = activation

        self.upconv = tf.keras.layers.Conv2DTranspose(filters // 2,
                                                      kernel_size=(self.pool_size, self.pool_size),
                                                      kernel_initializer=_get_kernel_initializer(filters,
                                                                                                 self.kernel_size),
                                                      strides=self.pool_size, padding=self.padding)

        self.activation_1 = tf.keras.layers.Activation(self.activation)

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.upconv(x)
        x = self.activation_1(x)
        return x

    def get_config(self):
        return dict(layer_idx=self.layer_idx,
                    filters_root=self.filters_root,
                    kernel_size=self.kernel_size,
                    pool_size=self.pool_size,
                    padding=self.padding,
                    activation=self.activation,
                    **super(UpconvBlock, self).get_config(),
                    )


class CropConcatBlock(tf.keras.layers.Layer):

    def call(self, inputs, **kwargs):
        processed, skipped = inputs
        skipped_shape = tf.shape(skipped)
        processed_shape = tf.shape(processed)

        height_diff = (skipped_shape[1] - processed_shape[1]) // 2
        width_diff = (skipped_shape[2] - processed_shape[2]) // 2
        assert tf.math.logical_and(tf.math.greater(height_diff, tf.constant(0, dtype=tf.int32)),
                                   tf.math.greater(width_diff, tf.constant(0, dtype=tf.int32)))

        skipped_cropped = skipped[:,
                          height_diff: (processed_shape[1] + height_diff),
                          width_diff: (processed_shape[2] + width_diff),
                          :]

        processed = tf.concat([skipped_cropped, processed], axis=-1)
        return processed


class ConvBlock(tf.keras.layers.Layer):

    def __init__(self, filters: List[int], kernel_size=3, dropout_rate=0.5, padding='valid', activation="relu",
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.padding = padding
        self.activation = activation
        self.convolutions = []
        self.dropouts = []
        self.activations = []

    def build(self, inputs):
        for filters in self.filters:
            self.convolutions.append(tf.keras.layers.Conv2D(filters=filters,
                                                            kernel_size=(self.kernel_size, self.kernel_size),
                                                            kernel_initializer=_get_kernel_initializer(filters,
                                                                                                       self.kernel_size),
                                                            strides=1,
                                                            padding=self.padding))
            self.dropouts.append(tf.keras.layers.Dropout(rate=self.dropout_rate))
            self.activations.append(tf.keras.layers.Activation(self.activation))

    def call(self, inputs, training=None, **kwargs):
        for i in range(len(self.convolutions)):
            inputs = self.convolutions[i](inputs)
            if training:
                inputs = self.dropouts[i](inputs)
            inputs = self.activations[i](inputs)
        return inputs

    def get_config(self):
        return dict(layer_idx=self.layer_idx,
                    filters_root=self.filters_root,
                    kernel_size=self.kernel_size,
                    dropout_rate=self.dropout_rate,
                    padding=self.padding,
                    activation=self.activation,
                    **super(ConvBlock, self).get_config(),
                    )


class UNetBlock(ResNetBlock):
    def __init__(self, layer, **kwargs):
        super(UNetBlock, self).__init__(layer, **kwargs)
        self.add_layer = CropConcatBlock()


class SimpleUNet(Submodel):
    def __init__(self, number_of_classes, filters_each_stage: list, name="U-Net",
                 **kwargs):
        self.classes = number_of_classes
        super(SimpleUNet, self).__init__(filters=filters_each_stage, name=name, **kwargs)

    def initialLayers(self) -> list:
        return []

    def coreLayers(self):
        save_the_filters = self.filters.copy()
        core = [ConvBlock(self.filters.pop(0)), self.coreSegment()]
        self.filters = save_the_filters
        return core

    def coreSegment(self):
        if len(self.filters) == 0:
            return Block([])
        current_filters = self.filters.pop(0)
        return UNetBlock(Block([tf.keras.layers.MaxPooling2D((2, 2)),
                                ConvBlock(current_filters),
                                self.coreSegment(),
                                UpconvBlock(current_filters[-1]),
                                ]))

    def finalizingLayers(self) -> list:
        return [tf.keras.layers.Conv2D(filters=self.classes,
                                       kernel_size=(1, 1),
                                       kernel_initializer=_get_kernel_initializer(self.filters[0][0], kernel_size=3),
                                       strides=1,
                                       padding='valid'),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Softmax(name="outputs")]


class UNet(SimpleUNet):
    def __init__(self, number_of_classes, base_filter, downscales: int):
        filters = self._get_filters(base_filter, downscales)
        super(UNet, self).__init__(number_of_classes=number_of_classes,
                                   filters_each_stage=filters)

    @staticmethod
    def _get_filters(base_filter, downscales):
        filters = list()
        for i in range(downscales):
            filters.append([base_filter, 2 * base_filter, 2 * base_filter])
            base_filter = 2 * base_filter
        return filters


class UNetGenerator(UNet):
    def __init__(self, number_of_classes, base_filter, downscales: int):
        super(UNetGenerator, self).__init__(number_of_classes, base_filter, downscales)

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
