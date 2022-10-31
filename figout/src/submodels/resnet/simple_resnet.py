from framework.submodel import Submodel, Block
import tensorflow as tf

from utils.layer_utilities import ResNetV2Block


class SimpleResNet(Submodel):
    def __init__(self, number_of_classes, filters_each_stage: list, blocks_per_stage: list = None, name="ResNet",
                 **kwargs):
        self.classes = number_of_classes
        if blocks_per_stage is None:
            self.blocks_per_stage = [3] * len(filters_each_stage)
        else:
            self.blocks_per_stage = blocks_per_stage
        assert len(blocks_per_stage) == len(filters_each_stage)-1
        super(SimpleResNet, self).__init__(filters=filters_each_stage, name=name, **kwargs)

    def initialLayers(self):
        return [tf.keras.layers.ZeroPadding2D(padding=(3, 3)),
                tf.keras.layers.Conv2D(self.filters[0], kernel_size=(7, 7), kernel_initializer='he_normal',
                                       strides=(2, 2), padding='valid'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.ZeroPadding2D(padding=(1, 1)),
                tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))
                ]

    @staticmethod
    def innerLayers(triple_filter, kernel_size=(3, 3), strides=(1, 1)):
        return [tf.keras.layers.Conv2D(triple_filter[0], kernel_size=(1, 1), kernel_initializer='he_normal',
                                       strides=strides),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(triple_filter[1], kernel_size=kernel_size,
                                       kernel_initializer='he_normal', padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(triple_filter[2], kernel_size=(1, 1), kernel_initializer='he_normal'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU()
                ]

    def identityBlock(self, triple_filter, kernel_size, **kwargs):
        if isinstance(triple_filter, int):
            triple_filter = [triple_filter, triple_filter, 4 * triple_filter]  # as in paper
        assert len(triple_filter) == 3
        main_block = Block()
        main_block.extend(self.innerLayers(triple_filter, kernel_size))
        return ResNetV2Block(main_block, name=kwargs['name'] + "Identity_block")

    def convolutionBlock(self, triple_filter, kernel_size, strides, **kwargs):
        if isinstance(triple_filter, int):
            triple_filter = [triple_filter, triple_filter, 4 * triple_filter]  # as in paper
        assert len(triple_filter) == 3
        main_block = Block()
        shortcut_block = Block()
        main_block.extend(self.innerLayers(triple_filter, kernel_size, strides))
        shortcut_block.extend([tf.keras.layers.Conv2D(triple_filter[-1], kernel_size=(1, 1),
                                                      kernel_initializer='he_normal',
                                                      strides=strides),
                               tf.keras.layers.BatchNormalization()])

        return ResNetV2Block(main_block, shortcut_block, name=kwargs['name'] + "Convolution_block")

    def coreSegment(self, single_filter, kernel_size, strides, stage_size, **kwargs):
        stage = [self.convolutionBlock(single_filter, kernel_size, strides, **kwargs)]
        for i in range(stage_size):
            stage.append(self.identityBlock(single_filter, kernel_size, **kwargs))

        return stage

    def coreLayers(self) -> list:
        core = []
        strides = [(1, 1)] + [(2, 2)] * len(self.filters[1:])
        for stage, (n_filters, stage_size, stride) in enumerate(zip(self.filters[1:], self.blocks_per_stage, strides)):
            core_block = self.coreSegment(n_filters, (3, 3), strides=stride, stage_size=stage_size,
                                          name="stage_{}_".format(stage))
            for block in core_block:
                core.append(block)
        return core

    def finalizingLayers(self):
        return [tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(self.classes, activation='softmax')]
