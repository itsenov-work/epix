from abc import ABC
from typing import List

import tensorflow as tf
import tensorflow.keras.applications as app
from framework.submodel import Block

"""https://keras.io/api/applications/"""


class PretrainedModel(Block):
    def __init__(self, frozen_percent=0, weights_source='imagenet', include_top=False, *args, **kwargs):
        """
        weights_source: str
            Where to load pretrained weights from? If 'imagenet', load pretraining on ImageNet,
            if system path - load weights from file at path, if None - random initialization (no pretraining)
        """
        super(PretrainedModel, self).__init__(**kwargs)
        self.model_args = args
        self.append(self._get_model(include_top, weights_source))
        self._remove_preprocessing()
        self.freeze_percent(frozen_percent)
        if weights_source is None:
            self.inference_mode = False
        else:
            self.inference_mode = True

    def _get_model(self, include_top, weights_source) -> tf.keras.models.Model:
        raise NotImplementedError

    def _remove_preprocessing(self):
        removed_layers = []
        self._remove_layers(removed_layers)

    def _remove_layers(self, layer_indexes: List):
        if len(layer_indexes) > 0:
            layer_indexes = sorted(layer_indexes)
            for idx in layer_indexes[::-1]:
                self.layers[0]._layers.pop(idx)

    def freeze_percent(self, percent):
        self.layers[0].trainable = False
        assert 0 <= percent <= 100
        n_frozen_layers = int((percent / 100.) * len(self.layers[0].layers))
        if n_frozen_layers < len(self.layers[0].layers):
            for layer in self.layers[0].layers[:n_frozen_layers:-1]:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True

    def call(self, inputs, **kwargs):
        kwargs['training'] = not self.inference_mode
        return super(PretrainedModel, self).call(inputs, **kwargs)


"""---EXAMPLES---"""


class ResNet50(PretrainedModel):
    def _get_model(self, include_top, weights_source):
        return app.ResNet50V2(include_top=include_top, weights=weights_source)


class EfficientNet(PretrainedModel, ABC):
    def _remove_preprocessing(self):
        # self._remove_layers([1, 2])
        pass


class EfficientNetB0(EfficientNet):
    def _get_model(self, include_top, weights_source):
        return app.EfficientNetB0(include_top=include_top, weights=weights_source)


class EfficientNetB1(EfficientNet):
    def _get_model(self, include_top, weights_source):
        return app.EfficientNetB1(include_top=include_top, weights=weights_source)


class EfficientNetB2(EfficientNet):
    def _get_model(self, include_top, weights_source):
        return app.EfficientNetB2(include_top=include_top, weights=weights_source)


class EfficientNetB3(EfficientNet):
    def _get_model(self, include_top, weights_source):
        return app.EfficientNetB3(include_top=include_top, weights=weights_source)


class EfficientNetB4(EfficientNet):
    def _get_model(self, include_top, weights_source):
        return app.EfficientNetB4(include_top=include_top, weights=weights_source)


class EfficientNetB5(EfficientNet):
    def _get_model(self, include_top, weights_source):
        return app.EfficientNetB5(include_top=include_top, weights=weights_source)


class EfficientNetB6(EfficientNet):
    def _get_model(self, include_top, weights_source):
        return app.EfficientNetB6(include_top=include_top, weights=weights_source)


class EfficientNetB7(EfficientNet):
    def _get_model(self, include_top, weights_source):
        return app.EfficientNetB7(include_top=include_top, weights=weights_source)
