import abc
import os
from _legacy.images.image_utilities import tensor_to_image
from utils.logger import LoggerMixin
import tensorflow as tf


class DataWriter(LoggerMixin, abc.ABC):
    def write(self, data):
        raise NotImplementedError


class FolderDataWriter(DataWriter, abc.ABC):
    logging_frequency = 100
    folder = None
    idx = None

    def write(self, data, data_names=None):
        if data_names is None:
            data_names = range(len(data))
        if len(data_names) != len(data):
            raise Exception("Names don't match data!")
        for idx, d in enumerate(data):
            self.idx = idx
            self.write_file(d, os.path.join(self.folder, str(data_names[idx])))

    def set_folder(self, folder):
        self.folder = folder

    def write_file(self, data, file_path):
        raise NotImplementedError


class ImageDataWriter(FolderDataWriter):
    def write_file(self, data, file_path):
        # Always save as png
        file_path_png = file_path + ".png"
        image = tensor_to_image(data)
        image = tf.image.encode_png(image)
        tf.io.write_file(file_path_png, image)
        idx = self.idx
        if (idx + 1) % self.logging_frequency == 0:
            self.log.i("%d images saved..." % (idx + 1))
