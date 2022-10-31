import tensorflow as tf
from models.gan import GAN


class ScamGAN(GAN):
    def train_step(self, data):
        batch_size = data.shape[0]
        if batch_size % 3 != 0:
            self.log.w("The batch size for ScamGAN must be divisible by 3. Current size: {}. Skipping step."
                       .format(batch_size))
            return

        artist_batch_size = data.shape[0] // 3
        artist_original_cards = data[:artist_batch_size, :, :, :]
        critic_original_cards = data[artist_batch_size:, :, :, :]

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            artist_fake_cards = self._get_fake_images(artist_original_cards)

            fake_input = tf.concat([artist_original_cards, artist_fake_cards], 0)
            real_output = self._get_real_decisions(critic_original_cards)
            fake_output = self._get_fake_decisions(fake_input)

            gen_loss = self.generator_loss(fake_output[:artist_batch_size, :])
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.artist.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.critic.trainable_variables)

        self.optimizers.artist_optimizer.apply_gradients(zip(gradients_of_generator, self.artist.trainable_variables))
        self.optimizers.critic_optimizer.apply_gradients(zip(gradients_of_discriminator, self.critic.trainable_variables))