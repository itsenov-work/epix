from abc import ABC
from typing import List
import tensorflow as tf

from tensorflow.keras.layers import Layer
from utils.logger import LoggerMixin


@tf.keras.utils.register_keras_serializable()
class Block(LoggerMixin, Layer):
    def __init__(self, layers: List[Layer] = None, *args, **kwargs):
        super(Block, self).__init__(*args, **kwargs)
        self.layers: List[Layer] = list()
        if layers is not None:
            self.extend(layers)

    def extend(self, l: List[Layer]):
        self.layers.extend(l)

    def append(self, layer):
        self.layers.append(layer)

    def call(self, inputs, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs, **kwargs)
        return inputs


class Submodel(tf.keras.layers.Layer, LoggerMixin, ABC):
    def __init__(self, filters, name="Submodel", **kwargs):
        super(Submodel, self).__init__(name=name, **kwargs)
        self.filters = filters

        self.initial_block = self.getBlock(layers=self.initialLayers(), name="InitialBlock")
        self.core_block = self.getBlock(layers=self.coreLayers(), name="CoreBlock")
        self.finalizing_block = self.getBlock(layers=self.finalizingLayers(), name="FinalizingBlock")

    def initialLayers(self) -> list:
        return []

    def coreSegment(self, *args) -> list:
        return []

    def coreLayers(self) -> list:
        core = []
        for n_filters in self.filters:
            core_block = self.coreSegment(n_filters, 3)
            for block in core_block:
                core.append(block)
        return core

    def finalizingLayers(self) -> list:
        return []

    @staticmethod
    def getBlock(layers, name='Block'):
        if isinstance(layers, Block):
            layers._name = name
            return layers
        elif isinstance(layers, list):
            return Block(layers=layers, name=name)

    def call(self, inputs, **kwargs):
        for block in (self.initial_block, self.core_block, self.finalizing_block):
            inputs = block(inputs, **kwargs)
        return inputs

    def predict(self, inputs):
        return self.call(inputs, training=False)

    def get_functional_model(self, input_layer: tf.keras.layers.Input):
        return tf.keras.models.Model(inputs=input_layer, outputs=self.call(input_layer), name=self.name)


