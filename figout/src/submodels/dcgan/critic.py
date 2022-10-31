from typing import List

import numpy as np

from framework.submodel import Submodel
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, Dense
from utils.layer_utilities import EqualizeLearningRate as ELR


class Critic(Submodel):
    def __init__(self, filters: List[int], dense_layers, name="Critic", **kwargs):
        self.dense_layers = dense_layers
        super(Critic, self).__init__(filters=np.array(list(reversed(filters))), name=name, **kwargs)

    def initialLayers(self):
        return []

    def coreSegment(self, single_filter, kernel_size):
        return [
            ELR(Conv2D(single_filter, kernel_size, strides=(1, 1), padding='same')),
            LeakyReLU(),
            Dropout(0.3)
        ]

    def finalizingLayers(self):
        final_layers = [Flatten()]
        for dense_layer in self.dense_layers:
            final_layers.append(Dense(dense_layer))
            final_layers.append(Dropout(0.5))
        final_layers.append(Dense(1))

        return final_layers

