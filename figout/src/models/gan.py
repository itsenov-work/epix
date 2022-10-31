import tensorflow as tf

from framework.compile import CompileConfig
from _legacy.framework.latent import LatentSpace
from framework.model import Model, ModelResults


class GANCompileConfig(CompileConfig):
    def __init__(self, discriminator_optimizer, generator_optimizer, loss_fn=None):
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.loss_fn = loss_fn


class GAN(Model):
    def __init__(self, artist, critic, latent: LatentSpace):
        super(GAN, self).__init__()

        self.submodels.artist = self.artist = artist
        self.submodels.critic = self.critic = critic
        self.latent = latent
        self.compile_config = None
        self.optimizers.artist_optimizer = None
        self.optimizers.critic_optimizer = None
        self.loss_fn = None

    def predict(self, inputs, **kwargs):
        return self.artist(inputs, training=False)

    def call(self, inputs, **kwargs):
        return self.artist(inputs)

    def compile(self, config: GANCompileConfig):
        self.compile_config = config
        self.optimizers.artist_optimizer = config.generator_optimizer
        self.optimizers.critic_optimizer = config.discriminator_optimizer
        loss_fn = config.loss_fn
        if loss_fn is None:
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss_fn = loss_fn
        self.store.set_checkpoint(tf.train.Checkpoint(generator_optimizer=self.artist_optimizer,
                                                      discriminator_optimizer=self.critic_optimizer,
                                                      generator=self.artist,
                                                      discriminator=self.critic))

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.loss_fn(tf.ones_like(fake_output), fake_output)

    def train_step(self, data):
        data_batch = data.get_batch()
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self._get_fake_images(batch_size=data_batch.shape[0])

            real_output = self._get_real_decisions(data_batch)
            fake_output = self._get_fake_decisions(generated_images)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.artist.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.critic.trainable_variables)

        self.optimizers.artist_optimizer.apply_gradients(zip(gradients_of_generator, self.artist.trainable_variables))
        self.optimizers.critic_optimizer.apply_gradients(zip(gradients_of_discriminator, self.critic.trainable_variables))

    def get_results(self, num_outputs):
        noise = self.latent.get_batch(num_outputs)
        return ModelResults([self.artist(noise, training=False)])

    def _get_fake_images(self, batch_size, training=True):
        noise = self.latent.get_batch(batch_size)
        return self.artist(noise, training=training)

    def _get_fake_decisions(self, generated_images, training=True):
        return self.critic(generated_images, training=training)

    def _get_real_decisions(self, cards, training=True):
        return self.critic(cards, training=training)

    def get_input_dimensions(self):
        return [self.artist.image_dims]

    def get_functional_models(self):
        # TODO: These are hardcoded
        artist_graph = self.artist.get_functional_model(tf.keras.layers.Input(shape=[None, self.artist.noise_dim]))
        critic_graph = self.critic.get_functional_model(tf.keras.layers.Input(shape=[None, *self.artist.image_dims]))
        return [artist_graph, critic_graph]
