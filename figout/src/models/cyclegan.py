from typing import List

import tensorflow as tf

from framework.compile import CompileConfig
from framework.model import Model, ModelResults

from framework.submodel import Submodel

from models.autoencoder import AutoEncodingLayer
from utils.layer_utilities import LeastSquareLoss


class CycleGANCompileConfig(CompileConfig):
    def __init__(self, generator_AtoB_optimizer=None, generator_BtoA_optimizer=None,
                 discriminator_A_optimizer=None, discriminator_B_optimizer=None,
                 loss_fn=None, loss_metric=None, accuracy_metric=None):

        self.generator_AtoB_optimizer = generator_AtoB_optimizer
        self.generator_BtoA_optimizer = generator_BtoA_optimizer
        self.discriminator_A_optimizer = discriminator_A_optimizer
        self.discriminator_B_optimizer = discriminator_B_optimizer
        self.loss_fn = loss_fn
        self.loss_metric = loss_metric
        self.accuracy_metric = accuracy_metric


class CycleGAN(Model):
    LAMBDA = 10

    def __init__(self, data, artists: List[Submodel], critics: List[Submodel], encoders: List[Submodel] = None,
                 **kwargs):

        super(CycleGAN, self).__init__(**kwargs)
        self._check_adversary(artists)
        self._check_adversary(critics)
        if encoders is not None:
            self._check_adversary(encoders)
            self.submodels.generator_AtoB = AutoEncodingLayer(encoders[0], artists[0])
            self.submodels.generator_BtoA = AutoEncodingLayer(encoders[1], artists[1])
        else:
            self.submodels.generator_AtoB = artists[0]
            self.submodels.generator_BtoA = artists[1]
        self.submodels.discriminator_A = critics[0]
        self.submodels.discriminator_B = critics[1]
        self.optimizers.generator_AtoB_optimizer = None
        self.optimizers.generator_BtoA_optimizer = None
        self.optimizers.discriminator_A_optimizer = None
        self.optimizers.discriminator_B_optimizer = None
        self.mae_loss = tf.losses.MeanAbsoluteError()  # L1
        self.mse_loss = tf.losses.MeanSquaredError()  # L2
        self.loss_obj = LeastSquareLoss()  # LS loss
        self.batch_of_A = data[0].get_batch()
        self.batch_of_B = data[1].get_batch()

    def _check_adversary(self, list_of_adv):
        if len(list_of_adv) != 2:
            self.log.e("The lists of artists, encoders, critics must contain two entries each")
            raise ValueError

    def compile(self, config: CycleGANCompileConfig):
        # initialize optimizers
        if config.generator_AtoB_optimizer is None:
            self.optimizers.generator_AtoB_optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.3)
        else:
            self.optimizers.generator_AtoB_optimizer = config.generator_AtoB_optimizer
        if config.generator_BtoA_optimizer is None:
            self.optimizers.generator_BtoA_optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.3)
        else:
            self.optimizers.generator_BtoA_optimizer = config.generator_BtoA_optimizer
        if config.discriminator_A_optimizer is None:
            self.optimizers.discriminator_A_optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.3)
        else:
            self.optimizers.discriminator_A_optimizer = config.discriminator_A_optimizer
        if config.discriminator_B_optimizer is None:
            self.optimizers.discriminator_B_optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.3)
        else:
            self.optimizers.discriminator_B_optimizer = config.discriminator_B_optimizer

    def discriminator_loss(self, real_output, fake_output):  # From GAN
        real_loss = self.loss_obj(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_obj(tf.zeros_like(fake_output), fake_output)
        total_loss = .5 * (real_loss + fake_loss)
        return total_loss

    def generator_loss(self, fake_output):  # From GAN
        return self.loss_obj(tf.ones_like(fake_output), fake_output)

    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.LAMBDA * 0.5 * loss

    def cycle_consistency_loss(self, real_image, cycled_image):
        loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return self.LAMBDA * loss

    def cycle_consistency(self, real_A, real_B):
        fake_B = self.submodels.generator_AtoB(real_A, training=True)
        cycled_A_B_A = self.submodels.generator_BtoA(fake_B, training=True)
        fake_A = self.submodels.generator_BtoA(real_B, training=True)
        cycled_B_A_B = self.submodels.generator_AtoB(fake_A, training=True)
        total_cycle_loss = self.cycle_consistency_loss(real_A, cycled_A_B_A) \
                           + self.cycle_consistency_loss(real_B, cycled_B_A_B)

        return total_cycle_loss

    def train_step(self, data: List[DataProvider]):
        with tf.GradientTape(persistent=True) as tape:  # persistent is because we use it more than once
            try:
                real_A = data[0].get_batch()
                real_B = data[1].get_batch()
            except ValueError as e:
                self.log.e("The dataset should be delivered to the train step as a list of two batches of images")

            # Cycle consistency loss:
            fake_B = self.submodels.generator_AtoB(real_A, training=True)
            cycled_A_B_A = self.submodels.generator_BtoA(fake_B, training=True)
            fake_A = self.submodels.generator_BtoA(real_B, training=True)
            cycled_B_A_B = self.submodels.generator_AtoB(fake_A, training=True)

            # Identity loss:
            same_A = self.submodels.generator_BtoA(real_A, training=True)
            same_B = self.submodels.generator_AtoB(real_B, training=True)

            # Objective losses:
            disc_real_A = self.submodels.discriminator_A(real_A, training=True)
            disc_real_B = self.submodels.discriminator_B(real_B, training=True)

            disc_fake_A = self.submodels.discriminator_A(fake_A, training=True)
            disc_fake_B = self.submodels.discriminator_B(fake_B, training=True)

            # calculate the losssssessssesesess
            gen_AtoB_loss = self.generator_loss(disc_fake_A)
            gen_BtoA_loss = self.generator_loss(disc_fake_B)

            total_cycle_loss = self.cycle_consistency_loss(real_A, cycled_A_B_A) \
                               + self.cycle_consistency_loss(real_B, cycled_B_A_B)

            identity_A_loss = self.identity_loss(real_A, same_A)
            identity_B_loss = self.identity_loss(real_B, same_B)

            # Get generator losses
            total_gen_AtoB_loss = gen_AtoB_loss + total_cycle_loss + identity_B_loss
            total_gen_BtoA_loss = gen_BtoA_loss + total_cycle_loss + identity_A_loss
            # Get discriminator losses
            disc_A_loss = self.discriminator_loss(disc_real_A, disc_fake_A)
            disc_B_loss = self.discriminator_loss(disc_real_B, disc_fake_B)

        # Calculate the gradients for generator and discriminator
        generator_AtoB_gradients = tape.gradient(total_gen_AtoB_loss,
                                                 self.submodels.generator_AtoB.trainable_variables)
        generator_BtoA_gradients = tape.gradient(total_gen_BtoA_loss,
                                                 self.submodels.generator_BtoA.trainable_variables)

        discriminator_A_gradients = tape.gradient(disc_A_loss,
                                                  self.submodels.discriminator_A.trainable_variables)
        discriminator_B_gradients = tape.gradient(disc_B_loss,
                                                  self.submodels.discriminator_B.trainable_variables)

        # Apply the gradients to the optimizer
        self.optimizers.generator_AtoB_optimizer.apply_gradients(zip(generator_AtoB_gradients,
                                                                     self.submodels.generator_AtoB.trainable_variables))

        self.optimizers.generator_BtoA_optimizer.apply_gradients(zip(generator_BtoA_gradients,
                                                                     self.submodels.generator_BtoA.trainable_variables))

        self.optimizers.discriminator_A_optimizer.apply_gradients(zip(discriminator_A_gradients,
                                                                      self.submodels.discriminator_A.trainable_variables))

        self.optimizers.discriminator_B_optimizer.apply_gradients(zip(discriminator_B_gradients,
                                                                      self.submodels.discriminator_B.trainable_variables))

    def get_results(self, num_outputs):
        data = []
        for i in range(num_outputs):
            image_A = self.batch_of_A[i]
            image_B = self.batch_of_B[i]
            data.append(
                [image_A, tf.squeeze(self.submodels.generator_AtoB.predict(tf.expand_dims(image_A, axis=0)), axis=0),
                 image_B, tf.squeeze(self.submodels.generator_BtoA.predict(tf.expand_dims(image_B, axis=0)), axis=0)])
        names = ['A', 'AtoB', 'B', 'BtoA']
        return ModelResults(data=data, names=names)
