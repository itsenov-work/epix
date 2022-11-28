from utils.augmentations import RandAugment, AutoAugment
import tensorflow as tf
from utils import logging


def _build_optimizer(learning_rate,
                     optimizer_name='rmsprop',
                     decay=0.9,
                     epsilon=0.001,
                     momentum=0.9):
    """Build optimizer.
  This is from EfficientNetV2 repo.
  """
    if optimizer_name == 'sgd':
        logging.info('Using SGD optimizer')
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == 'momentum':
        logging.info('Using Momentum optimizer')
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=momentum)
    elif optimizer_name == 'rmsprop':
        logging.info('Using RMSProp optimizer')
        optimizer = tf.keras.optimizers.RMSprop(learning_rate, decay, momentum,
                                                epsilon)
    elif optimizer_name == 'adam':
        logging.info('Using Adam optimizer')
        optimizer = tf.keras.optimizers.Adam(learning_rate)
    else:
        raise Exception('Unknown optimizer: %s' % optimizer_name)

    return optimizer


def _get_augmenter(flag_str):
    augmenter = None
    if flag_str == "auto":
        augmenter = AutoAugment()
    elif flag_str == "rand" or flag_str == "rand2":
        augmenter = RandAugment()
    elif flag_str == "rand1":
        augmenter = RandAugment(num_layers=1)
    elif flag_str == "rand3":
        augmenter = RandAugment(num_layers=3)

    return augmenter
