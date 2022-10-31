from abc import ABC

from submodels import pretrained_classifiers

from tensorflow.keras.layers import Dense, ReLU, Softmax, GlobalAveragePooling2D

from framework.submodel import Submodel


class PretrainedEfficientNet(Submodel, ABC):
    def __init__(self, n_classes, dense_layers, frozen_percent=0):
        self.dense_layers = dense_layers
        self.frozen_percent = frozen_percent
        self.n_classes = n_classes
        super(PretrainedEfficientNet, self).__init__(filters=None)

    def initialLayers(self):
        return []

    def coreLayers(self):
        raise NotImplementedError

    def finalizingLayers(self):
        fin = []
        for dense in self.dense_layers:
            fin.append(Dense(dense))
            fin.append(ReLU())
        fin.append(Dense(self.n_classes))
        fin.append(Softmax())
        return fin

    def get_config(self):
        return {"n_classes": self.n_classes,
                "dense_layers": self.dense_layers,
                "frozen_percent": self.frozen_percent
        }


class PretrainedEfficientNetB0(PretrainedEfficientNet):
    def coreLayers(self):
        return [pretrained_classifiers.EfficientNetB0(self.frozen_percent)]


class PretrainedEfficientNetB1(PretrainedEfficientNet):
    def coreLayers(self):
        return pretrained_classifiers.EfficientNetB1(self.frozen_percent)


class PretrainedEfficientNetB2(PretrainedEfficientNet):
    def coreLayers(self):
        return pretrained_classifiers.EfficientNetB2(self.frozen_percent)


class PretrainedEfficientNetB3(PretrainedEfficientNet):
    def coreLayers(self):
        return pretrained_classifiers.EfficientNetB3(self.frozen_percent)


class PretrainedEfficientNetB4(PretrainedEfficientNet):
    def coreLayers(self):
        return pretrained_classifiers.EfficientNetB4(self.frozen_percent)


class PretrainedEfficientNetB5(PretrainedEfficientNet):
    def coreLayers(self):
        return pretrained_classifiers.EfficientNetB5(self.frozen_percent)


class PretrainedEfficientNetB6(PretrainedEfficientNet):
    def coreLayers(self):
        return pretrained_classifiers.EfficientNetB6(self.frozen_percent)


class PretrainedEfficientNetB7(PretrainedEfficientNet):
    def coreLayers(self):
        return pretrained_classifiers.EfficientNetB7(self.frozen_percent)


class PretrainedEffNetV2B0(PretrainedEfficientNet):
    def coreLayers(self):
        return pretrained_classifiers.EffNetV2B0(self.frozen_percent)


class PretrainedEffNetV2B1(PretrainedEfficientNet):
    def coreLayers(self):
        return pretrained_classifiers.EffNetV2B1(self.frozen_percent)


class PretrainedEffNetV2B2(PretrainedEfficientNet):
    def coreLayers(self):
        return pretrained_classifiers.EffNetV2B2(self.frozen_percent)


class PretrainedEffNetV2B3(PretrainedEfficientNet):
    def coreLayers(self):
        return pretrained_classifiers.EffNetV2B3(self.frozen_percent)


class PretrainedEffNetV2small(PretrainedEfficientNet):
    def coreLayers(self):
        return pretrained_classifiers.EffNetV2small(self.frozen_percent)


class PretrainedEffNetV2medium(PretrainedEfficientNet):
    def coreLayers(self):
        return pretrained_classifiers.EffNetV2medium(self.frozen_percent)


class PretrainedEffNetV2large(PretrainedEfficientNet):
    def coreLayers(self):
        return pretrained_classifiers.EffNetV2large(self.frozen_percent)