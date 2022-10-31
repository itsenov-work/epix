import tensorflow as tf
from _legacy.data.data_properties import DimensionMixin
from framework.submodel import Submodel


class Encoder(Submodel):
    """Input:
        initial_dimension: original dimension of image (original_dim = height*width*channels)
        latent_dimension: desired intermediate dimension (dimension of product of encoder that goes to decoder)
        filters: list or array of number of filters for each layer, input for convolutional submodels.
    Confined variables:
        submodels = list of lists, each containing the needed submodels for a single convolution operation
    Output:
        coded data, after going through convolution and learning layer (Dense).
    """

    def __init__(self, filters, dense_layers, dimensions: DimensionMixin, noise_dimension, name='Encoder', **kwargs):
        self.dense_layers = dense_layers
        self.input_dims = dimensions.get_dimensions()
        self.noise_dim = noise_dimension
        self.final_conv_shape = None
        super(Encoder, self).__init__(filters=filters, name=name, **kwargs)

    def coreSegment(self, single_filter, kernel_size) -> list:
        return [tf.keras.layers.Conv2D(single_filter, kernel_size, strides=1, padding="same", name="EncoderConv2D"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.MaxPool2D((2, 2), name="EncoderMaxPool2D")
                ]

    def finalizingLayers(self) -> list:
        dense_list = [tf.keras.layers.Flatten()]
        for dense_layer in self.dense_layers[:-1]:
            dense_list.append(tf.keras.layers.Dense(dense_layer))
        dense_list.append(tf.keras.layers.Dense(self.noise_dim, name='latent'))
        return dense_list
