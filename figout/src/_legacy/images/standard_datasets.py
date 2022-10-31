import os

from utils.logger import LoggerMixin
import tensorflow_datasets as tfds
import tensorflow as tf


class StandardDatasets(LoggerMixin):
    """TODO: This should be split into:
        -provider for images only
        -provider for images with labels
        -provider for cyclegan images
        -providers for other data types"""
    img_folder = os.path.join("resources", "images", "standard_datasets")
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocessing_functions = list()
        self.dataset_info = None

    def provide(self, dataset_name, batch_size, labelled, split):
        self._check_dataset(dataset_name)
        dataset_folder = os.path.join(self.img_folder, dataset_name)

        builder = tfds.builder(dataset_name, data_dir=dataset_folder)
        builder.download_and_prepare(download_dir=dataset_folder)
        data = builder.as_dataset(split=split, shuffle_files=True, batch_size=batch_size, as_supervised=False)
        self.dataset_info = builder.info.features
        if not labelled:
            self.extract_image_only()
            self.convert_from_uint8()
            data = data.map(lambda x: self.apply_preproccessing(x))
        else:
            self.convert_from_uint8()
            data = data.map(lambda x: (self.apply_preproccessing(x['image']), x['label']))

        return data.repeat()

    def _check_dataset(self, name):
        catalog = tfds.list_builders()
        if name in catalog:
            self.log.i("Dataset found.")
        else:
            self.log.e("Dataset not available. Please choose from the following: \n" + str(catalog))

    @staticmethod
    def preprocess_dataset(dataset):
        dataset = dataset.map(lambda x: tf.image.convert_image_dtype(x, dtype=tf.float32))
        return dataset

    def convert_from_uint8(self):
        self.preprocessing_functions.append(lambda x: tf.image.convert_image_dtype(x, dtype=tf.float32))

    def extract_image_only(self):
        self.preprocessing_functions.append(lambda x: x['image'])

    def apply_preproccessing(self, image):
        for func in self.preprocessing_functions:
            image = func(image)
        return image
