import tensorflow as tf

import os

from keras.layers import BatchNormalization

from collect_data import tf_healthy_dataset
from models.regression import RegressionModel, RegressionCompileConfig
from submodels.dense import DenseNetwork


def _get_model_path_test():
    project_root = os.path.dirname(os.path.dirname(__file__))
    models_folder = os.path.join(project_root, "resources", "test")
    return os.path.join(models_folder, 'efficientnet', 'base_')


def plot_data(ds):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme()

    vals = np.fromiter(ds.map(lambda x, y: y), float)
    plt.hist(vals, bins=20)
    plt.title('Label Frequency')
    plt.show()
    plt.close()


if __name__ == '__main__':
    """NOTE: Error "Can't parse example... is because we fixed the length of cpgs, so change that number"""

    training_ds, test_ds = tf_healthy_dataset()
    training_ds = training_ds.map(
        lambda data: (data['data'], tf.expand_dims(data['age'], -1))
    )
    test_ds = test_ds.map(
        lambda data: (data['data'], tf.expand_dims(data['age'], -1))
    )

    # plot_data(training_ds)
    # plot_data(test_ds)
    training_ds = training_ds.repeat().batch(16)
    test_ds = test_ds.batch(16)

    training_ds = training_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    layers = [512, 512, 512, 512]

    normalization = BatchNormalization()

    regression = RegressionModel(
        submodel=DenseNetwork(filters=layers, dropout=0.5, preprocessing_layers=[], activation='elu'),
        output_dim=1
    )

    regression.compile(loss=RegressionCompileConfig().loss,
                       optimizer=RegressionCompileConfig().optimizer,
                       metrics=['mae'],
                       run_eagerly=False,
                       )
    # input_layer = tf.keras.Input(shape=[24993])
    # reg_fn = tf.keras.models.Model(inputs=input_layer, outputs=regression.call(input_layer))
    regression.fit(
        x=training_ds,
        y=None,
        validation_data=test_ds,
        epochs=200,
        steps_per_epoch=500
    )
