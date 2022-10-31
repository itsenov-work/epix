import tensorflow as tf
import numpy as np


class Decoder(tf.keras.layers.Layer):
    """The opposite of the Encoder function.
    """

    def __init__(self, dense_layers, convolution_dimension, filters, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.layers = []
        self.filters = np.array(list(reversed(filters)))[1:]
        self.dense_layers = np.array(list(reversed(dense_layers)))[1:]
        self.conv_dimension = convolution_dimension  # 2D latent space dimensions for convolution
        self.final_dense_dim = np.prod(convolution_dimension)
        self.createLayers()

    def createLayers(self):
        self.layers.append(self.makeDenseLayerGroup())
        # Add Convolutional Layers
        for filter_number in self.filters:
            self.layers.append(self.makeDeconvLayerGroup(filter_number, 3))  # using 3x3 kernel as default
        self.layers.append(self.makeFinalLayerGroup())

    @staticmethod
    def makeDeconvLayerGroup(filter_number, size_of_kernel):
        return [tf.keras.layers.Conv2DTranspose(filter_number, size_of_kernel, strides=2, padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(alpha=0.2)
                ]

    def makeDenseLayerGroup(self):
        dense_list = []
        for dense_layer in self.dense_layers:
            dense_list.append(tf.keras.layers.Dense(dense_layer))
        dense_list.append(tf.keras.layers.Dense(self.final_dense_dim))
        dense_list.append(tf.keras.layers.Reshape(self.conv_dimension))
        dense_list.append(tf.keras.layers.LeakyReLU(alpha=0.2))
        return dense_list

    @staticmethod
    def makeFinalLayerGroup():
        return [tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=2, padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("sigmoid", name="outputs")
                ]

    def call(self, latent_features):
        propagate = latent_features
        # Go through all submodels:
        for layer_group in self.layers:
            for layer in layer_group:
                propagate = layer(propagate)
        return propagate

