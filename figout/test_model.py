import random

import tensorflow as tf

from local_main import get_image_size
from models.classifier import Classifier
from tfx_components.model_builders import _get_efficientnet_model
import os

from tfx_components.periphery_builders import _build_optimizer


def _get_model_path_test():
    project_root = os.path.dirname(os.path.dirname(__file__))
    models_folder = os.path.join(project_root, "resources", "test")
    return os.path.join(models_folder, 'efficientnet', 'base_')


if __name__ == '__main__':
    seed = random.randint(0, 10000)
    image_size = get_image_size(0)

    path_to_drone_files = r"D:\yugigan project\ml data"
    insulator_files = os.path.join(path_to_drone_files, "INSULATOR classificator")
    commutation_files = os.path.join(path_to_drone_files, "Commutation classificator")
    pole_files = os.path.join(path_to_drone_files, "POLE classificator")
    dataset_files = insulator_files

    training_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_files,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=16)

    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_files,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=3)

    num_classes = len(training_ds.class_names)
    training_ds = training_ds.repeat()

    training_ds = training_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    validation_ds = validation_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # visualize_augments(training_ds, training_ds.class_names)

    training_ds = iter(training_ds)
    validation_ds = validation_ds

    efficient_net = _get_efficientnet_model(base=0, version=2)
    net = efficient_net(n_classes=4, dense_layers=[128], frozen_percent=70)
    model = Classifier(network=net)
    model_dir = _get_model_path_test()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=model_dir, update_freq='batch')
    savedmodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_dir, save_best_only=True)
    model.compile(optimizer=_build_optimizer(learning_rate=1e-5,
                                             optimizer_name='adam'),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy('labelling_accuracy', dtype=tf.float32),
                           tf.keras.metrics.SparseCategoricalCrossentropy('sparse_categorical_crossentropy',
                                                                          from_logits=False)])

    model.fit(
        training_ds,
        steps_per_epoch=100,
        validation_data=validation_ds,
        validation_steps=20,
        epochs=40,
        callbacks=[tensorboard_callback,
                   savedmodel_callback])