from dependency_injector.containers import DeclarativeContainer
from dependency_injector.providers import Configuration, DependenciesContainer, Factory, Callable, Singleton

from framework.training_schedule import TrainingSchedule, TrainRunner


class TrainingContainer(DeclarativeContainer):
    config = Configuration()
    data_container = DependenciesContainer()
    model_container = DependenciesContainer()
    train_schedule = Singleton(
        TrainingSchedule,
        model=model_container.model
    )
    runner = Singleton(
        TrainRunner,
        train_schedule=train_schedule,
        epochs=config.epochs,
        steps=config.steps,
        data=data_container.data_provider,
        batch_size=data_container.config.provided["data_provider"]["batch_size"]
    )