from keras.engine import compile_utils

import tensorflow as tf

from framework.submodel import Submodel


class Classifier(tf.keras.models.Model):
    def __init__(self, network: Submodel, **kwargs):
        super(Classifier, self).__init__(**kwargs)
        self.network = network
        self.config = None
        self.optimizer = None
        self.loss = None
        self.training_metrics = {}
        self.validation_metrics = {}

    def call(self, inputs, **kwargs):
        return self.network(inputs, **kwargs)

    def classification_loss(self, true_labels, predicted_labels):
        return self.compiled_loss(true_labels, predicted_labels)

    @tf.function(input_signature=[(tf.TensorSpec(shape=(None, None, None, None)),
                                   tf.TensorSpec(shape=(None,), dtype=tf.int32))])
    def train_step(self, data):
        images, true_labels = data
        with tf.GradientTape() as tape:
            labels = self(images, training=True)
            classification_loss = self.classification_loss(true_labels, labels)

        gradients = tape.gradient(classification_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        self.compiled_metrics.update_state(true_labels, labels)
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    @tf.function(input_signature=[(tf.TensorSpec(shape=(None, None, None, None)),
                                   tf.TensorSpec(shape=(None,), dtype=tf.int32))])
    def test_step(self, data):
        images, true_labels = data
        labels = self(images, training=False)
        classification_loss = self.classification_loss(true_labels, labels)
        self.compiled_metrics.update_state(true_labels, labels)
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    def get_config(self):
        return {"network": self.network}
