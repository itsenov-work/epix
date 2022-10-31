from typing import List

from _legacy.data.data_provider import SizeDataProviderMixin
from framework.callbacks.events import EpochEvent, StepEvent, Calendar, ModelTime, FunctionalEventMixin, PerStepEvent
from framework.model import Model
from submodels.progan.progressive_stage import ProgressiveStage
from models.gan import GAN
from utils.resize import Resize, ResizeMode


class EnterArtistStage(EpochEvent):
    def __init__(self, stage: ProgressiveStage, model: Model, epochs: int = 1):
        super(EnterArtistStage, self).__init__(model, epochs)
        self.stage = stage

    def trigger(self):
        self.log.debug(f"Triggering event {self.__class__.__name__} at {self.model.state.epoch} {self.model.state.step}")
        self.stage.active = True
        self.stage.in_transition = True
        self.stage.is_final = True
        self.stage.is_initial = False


class EnterCriticStage(EpochEvent):
    def __init__(self, stage: ProgressiveStage, model: Model, epochs: int = 1):
        super(EnterCriticStage, self).__init__(model, epochs)
        self.stage = stage

    def trigger(self):
        self.log.debug(f"Triggering event {self.__class__.__name__} at {self.model.state.epoch} {self.model.state.step}")
        self.stage.active = True
        self.stage.in_transition = True
        self.stage.is_initial = True
        self.stage.is_final = False


class ExitArtistStage(EpochEvent):
    def __init__(self, stage: ProgressiveStage, model: Model, epochs: int = 1):
        super(ExitArtistStage, self).__init__(model, epochs)
        self.stage = stage

    def trigger(self):
        self.log.debug(f"Triggering event {self.__class__.__name__} at {self.model.state.epoch} {self.model.state.step}")
        self.stage.is_final = False


class ExitCriticStage(EpochEvent):
    def __init__(self, stage: ProgressiveStage, model: Model, epochs: int = 1):
        super(ExitCriticStage, self).__init__(model, epochs)
        self.stage = stage

    def trigger(self):
        self.log.debug(f"Triggering event {self.__class__.__name__} at {self.model.state.epoch} {self.model.state.step}")
        self.stage.is_initial = False


class CompleteStage(StepEvent):
    def __init__(self, stage: ProgressiveStage, model: Model, epochs: int, steps: int = 200):
        super(CompleteStage, self).__init__(model, epochs, steps)
        self.stage = stage

    def trigger(self):
        self.log.debug(f"Triggering event {self.__class__.__name__} at {self.model.state.epoch} {self.model.state.step}")
        self.stage.in_transition = False


class UpsizeEvent(EpochEvent):
    def __init__(self, data_provider: SizeDataProviderMixin, size: Resize, model, epochs):
        super(UpsizeEvent, self).__init__(model, epochs)
        self.data_provider = data_provider
        self.size = size

    def trigger(self):
        self.log.debug(f"Triggering event {self.__class__.__name__} at {self.model.state.epoch} {self.model.state.step}")
        self.data_provider.set_size(self.size)


class AlphaIncrement(FunctionalEventMixin, PerStepEvent):
    pass


class ProgressiveCalendar(Calendar):
    def __init__(self,
                 model: GAN,
                 data_provider: SizeDataProviderMixin,
                 enter_times: List[ModelTime],
                 transition_steps: List[int]
                 ):
        super(ProgressiveCalendar, self).__init__(model)
        self.attach_artist_stages(model.artist.stages, enter_times, transition_steps)
        self.attach_critic_stages(model.critic.stages, enter_times, transition_steps)
        self.attach_data(data_provider, enter_times)

    # TODO: enter_times[0] and transition_steps[0] are not used
    def attach_artist_stages(self, stages: List[ProgressiveStage], enter_times: List[ModelTime], transition_steps: List[int]):
        stages[0].is_initial = True
        stages[0].active = True
        stages[0].is_final = True
        stages[0].in_transition = False

        epochs = 0
        trans_steps = 0
        for i in range(1, len(stages)):
            steps = transition_steps[i]
            epochs += enter_times[i].epochs
            trans_steps += steps

            self.append(ExitArtistStage(stage=stages[i - 1], model=self.model, epochs=epochs))
            self.append(EnterArtistStage(stage=stages[i], model=self.model, epochs=epochs))
            self.append(CompleteStage(model=self.model, stage=stages[i], epochs=epochs, steps=trans_steps))
            self.append(AlphaIncrement(function=stages[i].alpha.increment, steps=1, model=self.model))

    # TODO: enter_times[0] and transition_steps[0] are not used
    def attach_critic_stages(self, stages: List[ProgressiveStage], enter_times: List[ModelTime], transition_steps: List[int]):
        stages[0].is_initial = True
        stages[0].active = True
        stages[0].is_final = True
        stages[0].in_transition = False

        epochs = 0
        trans_steps = 0
        for i in range(1, len(stages)):
            steps = transition_steps[i]
            epochs += enter_times[i].epochs
            trans_steps += steps

            self.append(ExitCriticStage(stage=stages[i - 1], model=self.model, epochs=epochs))
            self.append(EnterCriticStage(stage=stages[i], model=self.model, epochs=epochs))
            self.append(CompleteStage(model=self.model, stage=stages[i], epochs=epochs, steps=trans_steps))
            self.append(AlphaIncrement(function=stages[i].alpha.increment, steps=1, model=self.model))

    def attach_data(self, data_provider: SizeDataProviderMixin, enter_times: List[ModelTime]):
        resolutions = self.model.artist.resolutions
        data_provider.set_size(Resize(resolutions[0][:2], mode=ResizeMode.FIXED))

        epochs = 0
        for i in range(1, len(resolutions)):
            r = resolutions[i]
            time = enter_times[i]
            epochs += time.epochs
            self.append(
                UpsizeEvent(data_provider=data_provider,
                            size=Resize(r[:2], ResizeMode.FIXED),
                            model=self.model,
                            epochs=epochs))