from tensorflow_addons.layers import InstanceNormalization
from framework.submodel import Submodel
import tensorflow as tf


class PatchCycleCritic(Submodel):
    def __init__(self, initial_filter: int, downsamples: int,
                 name="Cycle_Critic", **kwargs):
        self.initial_filter = initial_filter
        filters = self._create_filters(initial_filter, downsamples)
        self.final_filter = filters[-1]
        super(PatchCycleCritic, self).__init__(filters=filters[:-1], name=name, **kwargs)

    def initialLayers(self):
        return [tf.keras.layers.Conv2D(self.initial_filter, (4, 4), strides=2, padding='same'),
                tf.keras.layers.LeakyReLU(alpha=0.2)]

    def coreSegment(self, filter_number, size_of_kernel):
        return [tf.keras.layers.Conv2D(filter_number, 4, strides=2, padding='same', use_bias=False),
                InstanceNormalization(axis=-1),
                tf.keras.layers.LeakyReLU(alpha=0.2)
                ]

    def finalizingLayers(self):
        return [tf.keras.layers.Conv2D(self.final_filter, 4, strides=1, padding='same', use_bias=False),
                InstanceNormalization(axis=-1),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(1, 4, strides=1, padding='same', activation='tanh')
                ]

    @staticmethod
    def _create_filters(init_filter, downsamples):
        filters = [init_filter]
        for i in range(downsamples):
            filters.append(min(2 * filters[-1], 8 * filters[0]))
        return filters[1:]


class GlobalCycleCritic(PatchCycleCritic):
    def finalizingLayers(self):
        globalize = [tf.keras.layers.GlobalAveragePooling2D(),
                     tf.keras.layers.Dense(self.classes, activation='softmax')]
        super(GlobalCycleCritic, self).finalizingLayers() + globalize
