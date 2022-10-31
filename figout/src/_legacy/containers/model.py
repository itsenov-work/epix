from dependency_injector.containers import DeclarativeContainer
from dependency_injector.providers import Configuration, DependenciesContainer, Singleton, Selector, Container

from _legacy.containers import DCGANCompilatorContainer
from framework.compile import Compiler, CompileConfig
from _legacy.framework.latent import RandomNoiseLatentSpace

from models.aegan import AEGAN
from models.gan import GAN

from submodels.autoencoder.encoder import Encoder
from submodels.aegan.noise_critic import NoiseCritic
from submodels.dcgan.artist import Artist
from submodels.dcgan.critic import Critic


class DCGANContainer(DeclarativeContainer):
    config = Configuration()
    data_container = DependenciesContainer()
    latent = Singleton(
        RandomNoiseLatentSpace,
        config.latent.size
    )
    artist = Singleton(
        Artist,
        filters=config.artist.filters,
        dimensions=data_container.data_provider,
        noise_dimension=config.latent.size
    )
    critic = Singleton(
        Critic,
        filters=config.critic.filters,
        dense_layers=config.critic.dense_layers,
        # dimensions=data_container.data_provider,
        # noise_dimension=config.latent.size
    )
    model = Singleton(
        GAN,
        artist=artist,
        critic=critic,
        latent=latent
    )
    compile_config = DCGANCompilatorContainer.compile_config


class GANContainer(DeclarativeContainer):
    config = Configuration()
    data_container = DependenciesContainer()

    dcgan_container = Container(
        DCGANContainer,
        config=config,
        data_container=data_container
    )

    artist = Selector(
        config.type,
        dcgan=dcgan_container.artist
    )

    critic = Selector(
        config.type,
        dcgan=dcgan_container.critic
    )

    latent = Selector(
        config.type,
        dcgan=dcgan_container.latent
    )


class AEGANContainer(DeclarativeContainer):
    config = Configuration()
    data_container = DependenciesContainer()

    encoder = Singleton(
        Encoder,
        data_provider=data_container.data_provider,
        filters=config.encoder.filters,
        dense_layers=config.encoder.dense_layers,
        noise_dimension=config.gan.latent.size
    )

    gan_container = Container(
        GANContainer,
        config=config.gan,
        data_container=data_container
    )

    noise_critic = Singleton(
        NoiseCritic,
        noise_layers=config.noise_critic.noise_layers,
        decision_layers=config.noise_critic.decision_layers
    )
    model = Singleton(
        AEGAN,
        artist=gan_container.artist,
        critic=gan_container.critic,
        latent=gan_container.latent,
        encoder=encoder,
        noise_critic=noise_critic
    )
    compile_config = Singleton(
        CompileConfig
    )


class ModelContainer(DeclarativeContainer):
    config = Configuration()
    data_container = DependenciesContainer()

    dcgan_container = Container(
        DCGANContainer,
        config=config,
        data_container=data_container
    )

    aegan_container = Container(
        AEGANContainer,
        config=config,
        data_container=data_container
    )

    model = Selector(
        config.type,
        dcgan=dcgan_container.model,
        aegan=aegan_container.model
    )

    compile_config = Selector(
        config.type,
        dcgan=dcgan_container.compile_config,
        aegan=aegan_container.compile_config
    )

    compiler = Singleton(
        Compiler,
        model=model,
        config=compile_config
    )