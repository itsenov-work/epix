from dependency_injector.containers import DeclarativeContainer
from dependency_injector.providers import Configuration, Container

from _legacy.containers import CallbackContainer
from _legacy.containers import DataContainer
from _legacy.containers import ModelContainer
from _legacy.containers import TrainingContainer


class MainContainer(DeclarativeContainer):
    config = Configuration()

    data_container = Container(DataContainer,
                               config=config.data)
    model_container = Container(ModelContainer,
                                config=config.model,
                                data_container=data_container)
    callback_container = Container(CallbackContainer,
                                   config=config.callback,
                                   model_container=model_container,
                                   data_container=data_container)
    training_container = Container(TrainingContainer,
                                   config=config.training,
                                   model_container=model_container,
                                   data_container=data_container)