from abc import ABC
from typing import Callable, List

import os
from absl import flags

from framework.callbacks.callbacks import OurCallback
from utils.callback_utilities import WeightCheckpoint, WeightCheckpointManager, SavedModelCheckpoint
import tensorflow as tf

from utils.logger import LoggerMixin

FLAGS = flags.FLAGS


class ModelTime:
    def __init__(self, epochs=1, steps=0):
        self.epochs = epochs
        self.steps = steps


class Event(OurCallback, ABC):
    def trigger(self):
        raise NotImplementedError


class StepEvent(Event, ABC):
    def __init__(self, steps: int = 0):
        super(StepEvent, self).__init__()
        self.steps = steps

    def on_batch_begin(self, batch, logs=None):
        steps = batch
        if steps == self.steps:
            self.trigger()


class EpochEvent(Event, ABC):
    def __init__(self, epochs: int):
        super(EpochEvent, self).__init__()
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.epochs:
            self.trigger()


class PerEpochEvent(EpochEvent, ABC):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.epochs == 0:
            self.trigger()


class PerStepEvent(StepEvent, ABC):
    def __init__(self, steps: int):
        super(PerStepEvent, self).__init__(epochs=0, steps=steps)

    def on_batch_begin(self, batch, logs=None):
        steps = batch
        if steps % self.steps == 0:
            self.trigger()


class FunctionalEventMixin:
    def __init__(self, function: Callable, *args, **kwargs):
        super(FunctionalEventMixin, self).__init__(*args, **kwargs)
        self.function = function

    def trigger(self):
        self.function()


class Calendar(tf.keras.callbacks.Callback, List[Event]):
    def attach(self):
        for event in self:
            event.attach()


class SavingCallback(PerEpochEvent, ABC):
    subdir = 'model_save_type'

    def __init__(self, every_n_epochs=1):
        self.checkpoint = None
        self.checkpoint_manager = None
        self.path = None
        self.training_data = None
        self.validation_vata = None
        if every_n_epochs <= 0:
            every_n_epochs = 1
        super(SavingCallback, self).__init__(epochs=every_n_epochs)
        if every_n_epochs == 1:
            self.log.i("Model will be saved every epoch".format(every_n_epochs))
        else:
            self.log.i("Model will be saved every {}-th epoch".format(every_n_epochs))

    def fetch_extra_inputs(self, **kwargs):
        if 'training_data' in kwargs.keys():
            self.training_data = kwargs['training_data'].get_next()
        if 'validation_data' in kwargs.keys():
            self.validation_vata = kwargs['validation_data'].take(1)

    def on_train_begin(self, logs=None):
        model_folder = self.model.store.dirs().model_dir
        self.path = os.path.join(model_folder, "checkpoints", self.subdir)
        os.makedirs(self.path, exist_ok=True)
        self._set_checkpoint()
        self._set_manager()
        self.log.ok("Saving initialized.")

    def trigger(self):
        self.checkpoint_manager.save()
        self.log.ok("Thingy saved.")

    def _set_checkpoint(self):
        raise NotImplementedError

    def _set_manager(self):
        raise NotImplementedError


class WeightSavingCallback(SavingCallback):
    subdir = "model_weights"

    def _set_checkpoint(self):
        self.checkpoint = WeightCheckpoint(step_counter=self.model.state.step, model=self.model)

    def _set_manager(self):
        self.checkpoint_manager = WeightCheckpointManager(self.checkpoint, self.path, max_to_keep=3,
                                                          checkpoint_name=self.model.name)


class WeightRestoringCallback(WeightSavingCallback):
    """TODO: This needs testing still."""
    subdir = "model_weights"

    def __init__(self, flags: FLAGS, every_n_epochs=1):
        super(WeightRestoringCallback, self).__init__(every_n_epochs)
        self.flags = flags
        self.checkpoint = None
        self.checkpoint_manager = None

    def on_train_begin(self, logs=None):
        self._find_latest_weights(self.model, self.flags.flags_into_string())
        super(WeightRestoringCallback, self).on_train_begin(logs)
        if self.checkpoint_manager.latest_checkpoint is not None:
            # Calling once to get weights (this is how TF actually wants us to do this...)
            self.model.get_results(self.training_data)
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            self.log.i("Restored from {}".format(self.checkpoint_manager.latest_checkpoint))
        else:
            self.log.i("Initializing from scratch.")

    def _find_latest_weights(self, model, flagstr):
        class_path = model.store._dirs.model_class_dir
        subdirs = sorted(os.listdir(class_path))
        subdirs = [s for s in subdirs[::-1] if os.path.isdir(os.path.join(class_path, s))]
        for subdir in subdirs:
            path_to_flags = os.path.join(class_path, subdir, 'flags.txt')
            if os.path.isfile(path_to_flags):
                with open(path_to_flags, 'r') as f:
                    if flagstr == f.read():
                        model.store.dirs().model_dir = os.path.join(class_path, subdir)
                        self.path = os.path.join(class_path, subdir, "checkpoints", self.subdir)

                        self.log.i("Continuing the training run from " + subdir)
                        return
        path_to_flags = model.store.dirs().model_dir

        with open(os.path.join(path_to_flags, 'flags.txt'), 'w+') as f:
            f.write(flagstr)
        self.log.i("No previous weights were found. Starting training from scratch...")
        return


class ModelRestoringCallback(WeightRestoringCallback):
    subdir = "savedmodel"

    def __init__(self, flags: FLAGS, dataset, every_n_epochs=1):
        super(ModelRestoringCallback, self).__init__(flags, every_n_epochs)
        self.spec = dataset.element_spec

    def on_train_begin(self, logs=None):
        super(ModelRestoringCallback, self).on_train_begin()

    def _set_checkpoint(self):
        self.checkpoint = SavedModelCheckpoint(step_counter=self.model.state.step, model=self.model, spec=self.spec)
