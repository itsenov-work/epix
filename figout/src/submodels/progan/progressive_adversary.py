from abc import ABC
from typing import List

import tensorflow as tf

from tensorflow.keras.layers import Layer

from _legacy.data.data_properties import DimensionMixin
from framework.submodel import Submodel
from submodels.progan.progressive_stage import ProgressiveStage


class ProgressiveAdversary(ABC, Layer):
    def __init__(self, adversaries: List[Submodel], resolutions: List[DimensionMixin], **kwargs):
        super(ProgressiveAdversary, self).__init__(**kwargs)
        self.adversaries = adversaries
        self.stages = list()
        self.resolutions = [resolution.get_dimensions() for resolution in resolutions]
        for i in range(len(self.adversaries)):
            self.stages.append(self.create_stage(i))

    def create_stage(self, index: int) -> ProgressiveStage:
        raise NotImplementedError

    @staticmethod
    def factor_from_resolutions(smaller_res, larger_res):
        factors = [larger_res[i] / smaller_res[i] for i in range(2)]
        if not (factors[0].is_integer() and factors[0] == factors[1]):
            raise ValueError(f"Given resolutions: {larger_res}, {smaller_res} are not factors of each other!")
        return int(factors[0])

    def get_functional_model(self, input_layer: tf.keras.layers.Input):
        # TODO: Make a custom "Layer" class that does this
        return tf.keras.models.Model(inputs=input_layer, outputs=self.call(input_layer), name=self.name)