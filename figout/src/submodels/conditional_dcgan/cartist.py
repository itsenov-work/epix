from submodels.vanilla_artist import Artist
import tensorflow as tf


class ConditionedArtist(Artist):
    def __init__(self, filters, input_dims, n_classes, name="Conditional Artist"):
        super(ConditionedArtist, self).__init__(filters=filters,
                                                input_dims=input_dims,
                                                name=name)
        self.n_classes = n_classes
        self.label_path = []
        self.image_path = []
        self.merger = None

    def build(self, input_shapes):
        self.label_path.append(tf.keras.layers.Embedding(self.n_classes, 20))
        self.label_path.append(tf.keras.layers.GlobalAveragePooling1D())
        # self.label_path.append(tf.keras.submodels.Reshape((self.input_dims[0], self.input_dims[1], self.n_classes)))
        self.label_path.append(tf.keras.layers.Dense(self.input_dims[0] * self.input_dims[1]))
        self.label_path.append(tf.keras.layers.Reshape((self.input_dims[0], self.input_dims[1], 1)))
        self.image_path.append(tf.keras.layers.Dense(self.input_dims[0] * self.input_dims[1]))
        self.image_path.append(tf.keras.layers.Reshape((self.input_dims[0], self.input_dims[1], 1)))

        self.merger = tf.keras.layers.Concatenate()
        for n_filters in self.filters:
            self.makeConvLayerGroup(n_filters, 5)
        self.my_layers.append(tf.keras.layers.Conv2DTranspose(self.input_dims[2], 5, padding='same', activation=tf.keras.activations.tanh))

    @tf.function
    def call(self, data):
        noise, condition = data
        for cond_layer in self.label_path:
            condition = cond_layer(condition)
        for image_layer in self.image_path:
            noise = image_layer(noise)

        image = self.merger([noise, condition])
        for layer in self.my_layers:
            image = layer(image)

        return image

