import tensorflow as tf
from dependency_injector.containers import DeclarativeContainer
from dependency_injector.providers import Configuration, Factory

from models.gan import GANCompileConfig


class DCGANCompilatorContainer(DeclarativeContainer):
    config = Configuration()
    cross_entropy = Factory(
        tf.keras.losses.BinaryCrossentropy,
        from_logits=True
    )
    generator_optimizer = Factory(
        tf.keras.optimizers.Adam,
        lr=1e-3,
        beta_1=0.5
    )
    discriminator_optimizer = Factory(
        tf.keras.optimizers.Adam,
        lr=1e-3,
        beta_1=0.5
    )
    compile_config = Factory(
        GANCompileConfig,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        loss_fn=cross_entropy
    )
