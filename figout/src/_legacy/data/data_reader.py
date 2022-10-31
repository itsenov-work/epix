import os
from abc import ABC
from _legacy.images.image_utilities import image_file_to_tensor
import tensorflow as tf


class FolderDataReader(ABC):
    def get(self, folder):
        raise NotImplementedError


# For now, assume all images in a single folder
# Read image, decode and put into Dataset
# Automatically create TFRecord
# Second time reading images, read TFRecord directly

class ImageDataReader(FolderDataReader):
    def get(self, folder):
        files = os.listdir(folder)
        filepaths = [os.path.join(folder, file) for file in files]
        dataset = tf.data.Dataset.from_tensor_slices(filepaths)
        dataset = dataset.shuffle(len(filepaths))
        dataset = dataset.map(image_file_to_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat()
        return dataset


class LabelledImageDataReader(FolderDataReader):
    def get(self, folder):
        import os.path as osp

        for subfolder in os.listdir(folder):
            if not osp.isdir(osp.join(folder, subfolder)):
                continue
        files = os.listdir(folder)
        filepaths = [os.path.join(folder, file) for file in files]


        dataset = tf.data.Dataset.from_tensor_slices(filepaths)
        dataset = dataset.shuffle(len(filepaths))
        dataset = dataset.map(image_file_to_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat()
        return dataset