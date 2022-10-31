from submodels.vanilla_critic import Critic
import tensorflow as tf


class ConditionedCritic(Critic):
    def __init__(self, filters, dense_layers, input_dims, n_classes, name="Conditional Critic"):
        super(ConditionedCritic, self).__init__(filters=filters,
                                                dense_layers=dense_layers,
                                                name=name)
        self.input_dims = input_dims
        self.n_classes = n_classes
        self.label_path = []
        self.merger = None

    def build(self, input_shapes):
        self.label_path.append(tf.keras.layers.Embedding(self.n_classes, 50))
        self.label_path.append(tf.keras.layers.GlobalAveragePooling1D())  # try attention layer too
        self.label_path.append(tf.keras.layers.Dense(self.input_dims[0] * self.input_dims[1]))
        self.label_path.append(tf.keras.layers.Reshape((self.input_dims[0], self.input_dims[1], 1)))
        self.merger = tf.keras.layers.Concatenate()
        for n_filters in self.filters:
            self.makeConvLayerGroup(n_filters, 3)
        self.my_layers.append(tf.keras.layers.Flatten())
        for n_dense in self.dense_layers:
            self.my_layers.append(tf.keras.layers.Dense(n_dense))
            self.my_layers.append(tf.keras.layers.Dropout(0.5))
        self.my_layers.append(tf.keras.layers.Dense(1))

    @tf.function
    def call(self, data):
        image, condition = data
        for cond_layer in self.label_path:
            condition = cond_layer(condition)
        decision = self.merger([image, condition])
        for layer in self.my_layers:
            decision = layer(decision)
        return decision
