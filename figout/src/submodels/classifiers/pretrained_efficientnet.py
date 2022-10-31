from abc import ABC

from framework.submodel import Submodel
from submodels.classifiers import pretrained_classifiers

from tensorflow.keras.layers import Dense, ReLU, Softmax, GlobalAveragePooling2D


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
        fin = [GlobalAveragePooling2D()]
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
