import tensorflow as tf


class NoiseCritic(tf.keras.layers.Layer):
    def __init__(self, noise_layers, decision_layers, name="Noise_Critic", **kwargs):
        super(NoiseCritic, self).__init__(name=name, **kwargs)
        self.noise_path = []
        self.decision_path = []
        self.dense_layers_noise = noise_layers
        self.dense_layers_decision = decision_layers
        self.gaussian_noise = None
        self.concat_layer = None

    def build(self, input_shape):
        self.gaussian_noise = tf.keras.layers.GaussianNoise(0.01)
        for layer_size in self.dense_layers_noise:
            self.noise_path.append(tf.keras.layers.Dense(layer_size))
            self.noise_path.append(tf.keras.layers.GaussianDropout(0.005))
            self.noise_path.append(tf.keras.layers.LayerNormalization())
            self.noise_path.append(tf.keras.layers.LeakyReLU(0.02))
        self.concat_layer = tf.keras.layers.Concatenate()  # ([x, y])
        for layer_size in self.dense_layers_decision:
            self.decision_path.append(tf.keras.layers.Dense(layer_size))
            self.decision_path.append(tf.keras.layers.LeakyReLU(0.02))
        self.decision_path.append(tf.keras.layers.Activation('sigmoid'))
        self.decision_path.append(tf.keras.layers.Dense(1))

    def call(self, inputs, **kwargs):
        noise = self.gaussian_noise(inputs)
        processed_noise = noise
        for layer in self.noise_path:
            processed_noise = layer(processed_noise)
        decision = self.concat_layer([noise, processed_noise])
        for layer in self.decision_path:
            decision = layer(decision)

        return decision





