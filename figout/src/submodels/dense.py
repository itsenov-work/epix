import keras.layers

from framework.submodel import Submodel
import tensorflow as tf

class DenseNetwork(Submodel):
    i = 0

    def __init__(self, filters, activation='relu', dropout=0.5, preprocessing_layers=None):
        if preprocessing_layers is None:
            preprocessing_layers = []
        self.activation = activation
        self.dropout = dropout
        self.preprocessing_layers = preprocessing_layers
        super(DenseNetwork, self).__init__(filters=filters)

    def initialLayers(self) -> list:
        return self.preprocessing_layers

    def coreSegment(self, n_filters) -> list:
        self.i += 1
        return [
            keras.layers.Dense(n_filters, activation=self.activation, name=f"dense_{self.i}",
                               kernel_regularizer=tf.keras.regularizers.L2(l2=1e-3)),

            keras.layers.Dropout(self.dropout),
        ]

    def coreLayers(self) -> list:
        core = []
        for n_filters in self.filters:
            core_block = self.coreSegment(n_filters)
            for block in core_block:
                core.append(block)
        return core

