from submodels.resnet.simple_resnet import SimpleResNet
import tensorflow as tf


class GlobalResNetCritic(SimpleResNet):
    """ The ResNet Critic is exactly the same as a classifier, but with only 1 class, representing the decision"""

    def __init__(self, filters_each_stage: list, blocks_per_stage: list = None, name="ResNet Critic",
                 **kwargs):
        super(GlobalResNetCritic, self).__init__(number_of_classes=1,
                                                 filters_each_stage=filters_each_stage,
                                                 blocks_per_stage=blocks_per_stage,
                                                 name=name, **kwargs)


class PatchResNetCritic(SimpleResNet):
    def __init__(self, filters_each_stage: list, blocks_per_stage: list = None, name="ResNet Critic",
                 **kwargs):
        super(PatchResNetCritic, self).__init__(number_of_classes=1,
                                                filters_each_stage=filters_each_stage,
                                                blocks_per_stage=blocks_per_stage,
                                                name=name, **kwargs)

    def finalizingLayers(self):
        return [tf.keras.layers.Conv2D(1, kernel_size=7, strides=1, activation='tanh')]
