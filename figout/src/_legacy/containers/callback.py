from dependency_injector.containers import DeclarativeContainer
from dependency_injector.providers import Configuration, DependenciesContainer, Singleton

from framework.callbacks.callbacks import OneResultPerEpochCallback, GraphModelCallback


class CallbackContainer(DeclarativeContainer):
    config = Configuration()
    data_container = DependenciesContainer()
    model_container = DependenciesContainer()
    oneshots_callback = Singleton(
        OneResultPerEpochCallback,
        model=model_container.model,
        data_writer=data_container.data_writer
    )
    graph_callback = Singleton(
        GraphModelCallback,
        model=model_container.model,
    )
