from dependency_injector.containers import DeclarativeContainer
from dependency_injector.providers import Configuration, Singleton, Selector, Container, Dependency, DependenciesContainer

from _legacy.data.data_provider import EagerFolderProvider, LayerDataProvider, StandardTFDSProvider
from _legacy.data.data_reader import GrayScaleImageStoreDataReader, ImageStoreDataReader
from _legacy.data.data_writer import ImageDataWriter


class DataReaderContainer(DeclarativeContainer):
    """
    Different types of data readers.
    Select using data.data_reader.type:
        grayscale: A grayscale image reader
        rgb: An RGB image reader
    """
    config = Configuration()

    grayscale_reader = Singleton(
        GrayScaleImageStoreDataReader,
        size=config.size
    )

    rgb_reader = Singleton(
        ImageStoreDataReader,
        size=config.size,
    )


class DataProviderContainer(DeclarativeContainer):
    """
    Different types of data providers.
    Select using data.data_provider.type:
        eager_folder: An EagerFolder Provider
    """
    config = Configuration()

    data_reader = Dependency()
    eager_folder_provider = Singleton(
        EagerFolderProvider,
        data_reader=data_reader,
        data_folder=config.folder,
        batch_size=config.batch_size
    )
    standard_dataset_provider = Singleton(
        StandardTFDSProvider,
        dataset_name=config.dataset_name,
        batch_size=config.batch_size,
        labelled=config.labelled
    )


class DataContainer(DeclarativeContainer):
    """
    Main data container.
    Use DataContainer.data_reader, DataContainer.data_provider, DataContainer.data_writer
    """
    config = Configuration()

    data_reader_container = Container(
        DataReaderContainer,
        config=config.data_reader
    )
    data_reader = Selector(
        config.data_reader.type,
        grayscale=data_reader_container.grayscale_reader,
        rgb=data_reader_container.rgb_reader
    )

    data_provider_container = Container(
        DataProviderContainer,
        config=config.data_provider,
        data_reader=data_reader
    )

    data_provider = Selector(
        config.data_provider.type,
        eager_folder=data_provider_container.eager_folder_provider,
        standard_dataset=data_provider_container.standard_dataset_provider
    )

    data_writer = Singleton(ImageDataWriter)


class LayerDataContainer(DeclarativeContainer):
    layer = Dependency()
    data_container = DependenciesContainer()
    data_provider = Singleton(
        LayerDataProvider,
        layer=layer,
        data_provider=data_container.provided.data_provider
    )


class DownloadDataContainer(DeclarativeContainer):
    config = Configuration()

    data_container = DependenciesContainer()
    eager_folder_provider = Singleton(
        StandardTFDSProvider,
        batch_size=config.batch_size
    )



