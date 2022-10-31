import tensorflow as tf
from models.gan import GAN
from submodels.conditional_dcgan.cartist import ConditionedArtist
from submodels.conditional_dcgan.ccritic import ConditionedCritic
from numpy.random import randint
import numpy as np


class ConditionedGAN(GAN):
    def __init__(self, input_dimensions, n_classes, labeller=None, filters=None, dense_layers=None, noise_dim=None):
        super(ConditionedGAN, self).__init__(input_dimensions)

        self.artist = ConditionedArtist(filters=filters, input_dims=input_dimensions, n_classes=n_classes)
        self.critic = ConditionedCritic(filters=filters, input_dims=input_dimensions, n_classes=n_classes)
        self.labeller = labeller
        self.n_classes = n_classes

    def predict(self, inputs):
        return self.artist(inputs[0], inputs[1], training=False)

    def _get_fake_images(self, cards):
        noise = self.make_noise(cards)
        return [self.artist(noise[0], noise[1], training=True), noise[1]]

    def _get_fake_decisions(self, generated_images):
        return self.critic(generated_images[0], generated_images[1], training=True)

    def _get_real_decisions(self, cards):
        return self.critic(cards[0], cards[1], training=True)

    def make_noise(self, cards):
        return self._noisy_boy(tf.shape(cards[1])[0], self.noise_dim)

    def _noisy_boy(self, batch_size, amount):
        if self.labeller is not None:
            labels = self.labeller.transform(self._generate_attr_label(batch_size))
            # labels = self.labeller.transform(np.c_[self._generate_race_label(batch_size),
            #                                        self._generate_attr_label(batch_size)])
        else:
            labels = tf.expand_dims((2 ** np.random.randint(0, self.n_classes, size=batch_size)).astype(float), axis=-1)
        noise = tf.random.normal([batch_size, amount])
        return [noise, labels]  # or make into tuple? which is better?

