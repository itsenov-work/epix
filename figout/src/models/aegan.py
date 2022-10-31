import tensorflow as tf
from _legacy.framework.latent import LatentSpace
from framework.model import ModelResults
from framework.submodel import Submodel
from submodels.autoencoder.encoder import Encoder
from submodels.aegan.noise_critic import NoiseCritic
from models.gan import GAN


class AEGAN(GAN):
    def __init__(self,
                 artist: Submodel,
                 critic: Submodel,
                 latent: LatentSpace,
                 encoder: Encoder,
                 noise_critic: NoiseCritic,
                 data):
        super(AEGAN, self).__init__(artist, critic, latent)
        self.submodels.encoder = self.encoder = encoder
        self.submodels.image_critic = self.image_critic = critic
        self.submodels.noise_critic = self.noise_critic = noise_critic
        self.optimizers.image_critic_optimizer = self.optimizers.noise_critic_optimizer = self.optimizers.aegan_optimizer = None
        self.loss_fn = self.mae_loss = self.mse_loss = None
        self.batch_of_data = data.get_batch()

    def compile(self, *args):
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mae_loss = tf.keras.losses.MeanAbsoluteError()
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.optimizers.noise_critic_optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        self.optimizers.image_critic_optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        self.optimizers.aegan_optimizer = tf.keras.optimizers.Adam(lr=2e-4)

    def call(self, x_real, z_real):
        z_hat = self.encoder(x_real)
        x_hat = self.artist(z_real)
        x_tilde = self.artist(z_hat)
        z_tilde = self.encoder(x_hat)
        prediction_x_hat = self.image_critic(x_hat, training=False)
        prediction_x_tilde = self.image_critic(x_tilde, training=False)
        prediction_z_hat = self.noise_critic(z_hat, training=False)
        prediction_z_tilde = self.noise_critic(z_tilde, training=False)

        return x_tilde, z_tilde, prediction_x_hat, prediction_x_tilde, prediction_z_hat, prediction_z_tilde

    def predict(self, noise):
        return self.artist(noise, training=False)

    def train_step(self, data):
        batch_size = data.get_batch().shape[0]

        with tf.GradientTape(persistent=True) as adversarial_loss_tape:
            images_from_noise = self._get_fake_images(batch_size, training=False)  # G(z)
            images_from_latent = self._get_ae_output(data.get_batch(), training=False)  # G(E(x))
            real_output_noise_1 = self.image_critic(data.get_batch(), training=True)  # D_x(x)
            fake_output_noise_1 = self.image_critic(images_from_noise, training=True)  # D_x(G(z))
            noise_loss_image_critic = self.discriminator_loss(real_output_noise_1, fake_output_noise_1)  # (2)

            real_output_noise_2 = self.image_critic(data.get_batch(), training=True)  # D_x(x)
            fake_output_noise_2 = self.image_critic(images_from_latent, training=True)  # D_x(G(E(x))
            latent_loss_image_critic = self.discriminator_loss(real_output_noise_2, fake_output_noise_2)  # (3)

            # image_critic_loss = noise_loss_image_critic + latent_loss_image_critic  # (2) + (3)

            encoded_real = self.encoder(data.get_batch(), training=False)  # E(x)
            encoded_fake = self._get_reverse_ae_output(self.latent.get_batch(batch_size), training=False)  # E(G(z))
            real_output_noise_critic_1 = self.noise_critic(self.latent.get_batch(batch_size), training=True)  # D_z(z)
            fake_output_noise_critic_1 = self.noise_critic(encoded_real, training=True)  # D_z(E(x))
            noise_loss_noise_critic = self.discriminator_loss(real_output_noise_critic_1,
                                                              fake_output_noise_critic_1)  # (4)
            real_output_noise_critic_2 = self.noise_critic(self.latent.get_batch(batch_size), training=True)  # D_z(z)
            fake_output_noise_critic_2 = self.noise_critic(encoded_fake, training=True)  # D_z(E(G(z)))
            latent_loss_noise_critic = self.discriminator_loss(real_output_noise_critic_2,
                                                               fake_output_noise_critic_2)  # (5)

            # noise_critic_loss = noise_loss_noise_critic + latent_loss_noise_critic  # (4) + (5)

        gradients_of_image_discriminator_2 = adversarial_loss_tape.gradient(noise_loss_image_critic,
                                                                            self.image_critic.trainable_variables)
        gradients_of_image_discriminator_3 = adversarial_loss_tape.gradient(latent_loss_image_critic,
                                                                            self.image_critic.trainable_variables)
        gradients_of_noise_discriminator_4 = adversarial_loss_tape.gradient(noise_loss_noise_critic,
                                                                            self.noise_critic.trainable_variables)
        gradients_of_noise_discriminator_5 = adversarial_loss_tape.gradient(latent_loss_noise_critic,
                                                                            self.noise_critic.trainable_variables)

        self.optimizers.image_critic_optimizer.apply_gradients(
            zip(gradients_of_image_discriminator_2, self.image_critic.trainable_variables))
        self.optimizers.image_critic_optimizer.apply_gradients(
            zip(gradients_of_image_discriminator_3, self.image_critic.trainable_variables))
        self.optimizers.noise_critic_optimizer.apply_gradients(
            zip(gradients_of_noise_discriminator_4, self.noise_critic.trainable_variables))
        self.optimizers.noise_critic_optimizer.apply_gradients(
            zip(gradients_of_noise_discriminator_5, self.noise_critic.trainable_variables))

        for j in range(4):
            with tf.GradientTape(persistent=False) as reconstruction_loss_tape:
                x = data.get_batch()
                z = self.latent.get_batch(batch_size)
                g_e_x, e_g_z, dx_g_z, dx_g_e_x, dz_e_x, dz_e_g_z = self(x, z)
                L_g_e_x = self.mse_loss(x, g_e_x)  # ||G(E(x)) - x||_1
                L_e_g_z = self.mae_loss(z, e_g_z)  # ||E(G(z)) - z||_2
                L_dx = self.discriminator_loss(dx_g_z, dx_g_e_x)
                L_dz = self.discriminator_loss(dz_e_x, dz_e_g_z)

                total_aegan_loss = L_g_e_x + L_e_g_z + L_dx + L_dz

            gradients_of_aegan = reconstruction_loss_tape.gradient(total_aegan_loss, self.trainable_variables)
            self.optimizers.aegan_optimizer.apply_gradients(zip(gradients_of_aegan, self.trainable_variables))

    def ae_loss(self, inputs, training=True):
        image = self._get_ae_output(inputs, training=training)
        return self.mse_loss(inputs, image)

    def reverse_ae_loss(self, noise, training=True):
        latent = self._get_reverse_ae_output(noise, training=training)
        return self.mae_loss(noise, latent)

    def _get_ae_output(self, inputs, training=True):
        latent = self.encoder(inputs, training=training)
        return self.artist(latent, training=training)

    def _get_reverse_ae_output(self, inputs, training=True):
        image = self.artist(inputs, training=training)
        return self.encoder(image, training=training)

    def get_input_dimensions(self):
        return [self.artist.image_dims, self.encoder.noise_dim]

    def get_functional_models(self):
        noise = tf.keras.layers.Input(shape=self.encoder.noise_dim)
        image = tf.keras.layers.Input(shape=self.artist.image_dims)
        artist_graph = self.artist.get_functional_model(noise)
        image_critic_graph = self.image_critic.get_functional_model(image)
        noise_critic_graph = None
        encoder_graph = self.encoder.get_functional_model(image)
        main_graph = tf.keras.models.Model(inputs=[image, noise], outputs=self.call(image, noise))

        return [artist_graph, image_critic_graph, encoder_graph, main_graph]

    def get_results(self, num_outputs):
        images = self.batch_of_data[:num_outputs]
        data = [images, self._get_ae_output(images, training=False), self._get_fake_images(num_outputs)]
        names = ['original', 'autoencoded', 'from_noise']

        return ModelResults(data=data, names=names)

