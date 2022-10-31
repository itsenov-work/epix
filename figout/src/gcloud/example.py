from abc import ABC
import os.path as osp
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from pathvalidate import sanitize_filename
from tqdm import tqdm


class DataFeature(ABC):
    def __new__(cls, data):
        return cls.create(data=data)

    @classmethod
    def create(cls, data):
        raise NotImplementedError


class ImageFeature(DataFeature):
    @classmethod
    def create(cls, data: tf.Tensor):
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.encode_png(data).numpy()])
        )


class StringFeature(DataFeature):
    @classmethod
    def create(cls, data: str):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.encode()]))


class FloatFeature(DataFeature):
    @classmethod
    def create(cls, data):
        if isinstance(data, float):
            return tf.train.Feature(float_list=tf.train.FloatList(value=[data]))
        elif isinstance(data, tf.Tensor):
            return tf.train.Feature(float_list=tf.train.FloatList(value=[data.numpy()]))
        elif isinstance(data, list):
            return tf.train.Feature(float_list=tf.train.FloatList(value=data))


class IntFeature(DataFeature):
    @classmethod
    def create(cls, data):
        if isinstance(data, int):
            return tf.train.Feature(float_list=tf.train.Int64List(value=[data]))
        elif isinstance(data, tf.Tensor):
            return tf.train.Feature(float_list=tf.train.Int64List(value=[data.numpy()]))
        elif isinstance(data, list):
            return tf.train.Feature(float_list=tf.train.Int64List(value=data))


class ExampleEncoder(ABC):
    def __init__(self, name):
        self.name = name
        self.folder_name = sanitize_filename(self.name)

    def encode(self, data) -> tf.train.Example:
        raise NotImplementedError

    def write(self, dataset: tf.data.Dataset, dataset_info, examples_per_file: int = 300):
        os.makedirs(self.tfrecords_folder, exist_ok=True)
        i = 0
        iterator = iter(dataset)
        while True:
            tfrecord_file = osp.join(self.tfrecords_folder, f'{i}.tfrecords')
            writer = tf.io.TFRecordWriter(tfrecord_file)
            print(f"Writing file {tfrecord_file}:")
            for i in tqdm(range(examples_per_file)):
                data = next(iterator, None)
                if data is None:
                    return
                example = dataset_info.features.encode_example({
                    key: value.numpy() for key, value in data.items()
                })
                example_str = example.SerializeToString()
                writer.write(example_str)

    @staticmethod
    def get_examples_folder():
        from utils.dir import Dir
        return osp.join(Dir.get_resources_dir(), "examples")

    @property
    def tfrecords_folder(self):
        return osp.join(self.get_examples_folder(), self.folder_name)


class OldLabelledImageExampleEncoder(ExampleEncoder):
    def encode(self, data) -> tf.train.Example:
        image = data[0]
        label = data[1]
        return tf.train.Example(
            features=tf.train.Features(feature={
                "image": ImageFeature(image),
                "label": IntFeature(label),
            })
        )


class LabelledImageExampleEncoder(ExampleEncoder):
    def encode(self, data) -> tf.train.Example:
        return tfds.features.FeatureConnector().encode_example(data)

