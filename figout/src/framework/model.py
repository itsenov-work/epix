from abc import ABC
from threading import RLock
from typing import List

import tensorflow as tf

from framework.model_state import ModelState
from framework.model_store import ModelStore
from utils.easydict import EasyDict
from utils.logger import LoggerMixin


class ModelResults(object):
    def __init__(self, data, names: List[str] = None):
        self.data = data
        self.names = names

    def add_results(self, new_MR):
        self.data.extend(new_MR.data)
        self.names.extend(new_MR.names)

    def __len__(self):
        return len(self.data)


class Model(LoggerMixin, tf.keras.models.Model, ABC):
    def __init__(self, **kwargs):
        """
        Note: All arguments should be passed as keywords from the implementation of this class.
        They will be saved as a  dictionary in self.arguments to be used for identification
        """
        super(Model, self).__init__(**kwargs)

        self.arguments = kwargs
        # Model state
        self.state = ModelState(self)
        # Model store
        self.store = ModelStore(self)
        self.submodels = EasyDict()
        self.optimizers = EasyDict()

    def get_results(self, num_outputs) -> ModelResults:
        raise NotImplementedError

    @tf.function
    def train_step(self, data):
        raise NotImplementedError

    @tf.function
    def test_step(self, data):
        raise NotImplementedError

    def compile_config(self, config):
        raise NotImplementedError

