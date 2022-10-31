import tensorflow as tf
from tensorflow.keras.layers import Layer

from framework.compile import CompileConfig
from framework.model import Model
from framework.submodel import Submodel


class AutoEncodingLayer(Layer):
    def __init__(self, encoder: Submodel, decoder: Submodel):
        super(AutoEncodingLayer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def predict(self, inputs, **kwargs):
        inputs = self.encoder(inputs, training=False, **kwargs)
        return self.decoder(inputs, training=False, **kwargs)

    def call(self, inputs, **kwargs):
        inputs = self.encoder(inputs, **kwargs)
        return self.decoder(inputs, **kwargs)


class AECompileConfig(CompileConfig):
    def __init__(self, optimizer, loss_fn=None):
        self.optimizer = optimizer
        self.loss_fn = loss_fn


class AutoEncoder(Model, AutoEncodingLayer):
    def __init__(self, encoder: Submodel, decoder: Submodel):
        super(AutoEncodingLayer, self).__init__(encoder, decoder)
        self.submodels.encoder = self.encoder
        self.submodels.decoder = self.decoder
        self.optimizers.optimizer = None
        self.loss_fn = None

    def compile(self, config: AECompileConfig):
        self.optimizers.optimizer = config.optimizer
        loss_fn = config.loss_fn
        if loss_fn is None:
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss_fn = loss_fn

    def reconstruction_loss(self, inputs):
        reconstructed = self.call(inputs)
        return self.loss_fn(inputs, reconstructed)

    def train_step(self, data):
        data_batch = data.get_batch()
        with tf.GradientTape() as tape:
            reconstruction_loss = self.reconstruction_loss(data_batch)

        gradients = tape.gradient(reconstruction_loss, self.trainable_variables)
        self.optimizers.optimizer.apply_gradients(zip(gradients, self.trainable_variables))




