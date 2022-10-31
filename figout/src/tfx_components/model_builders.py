import os

import keras_tuner
from absl import flags
import tensorflow as tf
from keras_tuner import HyperModel

import submodels.pretrained_efficientnet as effnets
from models.classifier import Classifier
from tfx_components.periphery_builders import _build_optimizer
from utils.logger import LoggerMixin

FLAGS = flags.FLAGS


def _get_efficientnet_model(base, version=1):
    base_dict_v1 = {0: effnets.PretrainedEfficientNetB0,
                    1: effnets.PretrainedEfficientNetB1,
                    2: effnets.PretrainedEfficientNetB2,
                    3: effnets.PretrainedEfficientNetB3,
                    4: effnets.PretrainedEfficientNetB4,
                    5: effnets.PretrainedEfficientNetB5,
                    6: effnets.PretrainedEfficientNetB6,
                    7: effnets.PretrainedEfficientNetB7}

    # TODO: Add small, medium, large (already accessible)
    base_dict_v2 = {0: effnets.PretrainedEffNetV2B0,
                    1: effnets.PretrainedEffNetV2B1,
                    2: effnets.PretrainedEffNetV2B2,
                    3: effnets.PretrainedEffNetV2B3,
                    }

    if version == 1:
        return base_dict_v1[base]
    elif version == 2:
        return base_dict_v2[base]
    else:
        return ModuleNotFoundError


class FixedModel(LoggerMixin, HyperModel):
    """ Object that searches for previously saved model, and if it does not exist, creates one and returns it"""

    def __init__(self, flags: dict, n_classes: int, model_root: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flags = flags
        self.n_classes = n_classes
        self.model_root = model_root

    def get_hyperparameters(self) -> kerastuner.HyperParameters:
        """Returns hyperparameters for building Keras model."""
        hp = kerastuner.HyperParameters()
        hp.Int('base', min_value=0, max_value=7, default=self.flags['base'])
        hp.Float('learning_rate', min_value=1e-8, max_value=1e-3, default=self.flags['lr'],
                 sampling="log")
        hp.Choice('optimizer', values=['sgd, ''momentum', 'rmsprop', 'adam'], default=self.flags['optimizer'])
        hp.Int('freeze_percent', min_value=0, max_value=100, default=self.flags['freeze_percent'])
        hp.Int('dense_layer_number', min_value=1, max_value=4, default=1)
        hp.Choice('initial_dense_layer', values=[2 ** n for n in range(5, 11)], default=128)
        return hp

    def _build_model(self, hp):
        efficient_net = _get_efficientnet_model(hp.get("base"))
        net = efficient_net(n_classes=self.n_classes, dense_layers=[128], frozen_percent=hp.get("freeze_percent"))
        return Classifier(network=net)

    @staticmethod
    def _compile_model(model, hp):
        # TODO: how do you know if a copy is created, i.e., can you just compile not return anything in this function
        model.compile(optimizer=_build_optimizer(learning_rate=hp.get('learning_rate'),
                                                 optimizer_name=hp.get('optimizer')),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy('labelling_accuracy', dtype=tf.float32)
                               ]
                      )

        return model

    def build(self, hp) -> tf.keras.models.Model:
        path_to_savedmodel = os.path.join(self.model_root, "checkpoints", "savedmodel")
        if os.path.exists(path_to_savedmodel):
            try:
                model = tf.saved_model.load(path_to_savedmodel)
                self.log.ok("SavedModel found and loaded")
                if model._is_compiled is False:
                    model = self._compile_model(model, hp)
                return model
            except FileNotFoundError:
                self.log.i("No previously saved model. Creating from scratch.")
                os.makedirs(path_to_savedmodel, exist_ok=True)
                model = self._build_model(hp)
                model = self._compile_model(model, hp)
                return model


class TunedModel(FixedModel):

    @staticmethod
    def _get_top_layers_arg(layer_number, initial):
        # A rather simple logic of each subsequent layer having 1/2 nodes.
        return [int(initial / n) for n in range(1, layer_number + 1)]

    def build(self, hp) -> tf.keras.models.Model:
        efficient_net = _get_efficientnet_model(hp.get('base'))
        net = efficient_net(n_classes=self.n_classes,
                            dense_layers=self._get_top_layers_arg(hp.get('dense_layer_number'),
                                                                  hp.get('initial_dense_layer')),
                            frozen_percent=hp.get('freeze_percent'))
        model = Classifier(network=net)
        model = self._compile_model(model, hp)
        return model
