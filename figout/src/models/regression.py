import tensorflow.keras as keras

from framework.compile import CompileConfig
from framework.model import Model
from framework.submodel import Submodel
import tensorflow as tf


class RegressionCompileConfig(CompileConfig):
    def __init__(self, loss='mean_absolute_error', optimizer=keras.optimizers.Adam(1e-4)):
        self.loss = loss
        self.optimizer = optimizer


class RegressionModel(tf.keras.Model):
    def __init__(self, submodel: Submodel, output_dim: int, activation='relu'):
        super(RegressionModel, self).__init__()
        self.submodel = submodel
        self.output_dim = output_dim
        self.final_layer = keras.layers.Dense(units=output_dim, activation=activation)

    def call(self, inputs):
        inputs = self.submodel(inputs)
        return self.final_layer(inputs)
