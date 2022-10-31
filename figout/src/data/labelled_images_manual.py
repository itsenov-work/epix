import os
import os.path as osp
from typing import List, Tuple
import random
import tensorflow as tf
from tensorflow import Tensor

IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp', 'gif']


def serialize_images_to_tfrecord(
        dataset_name: str,
        root_folder: str,
        tfrecords_folder: str,
        image_size: Tuple[int, int],
        images_per_tfrecord: int = 100,
        shuffle_seed: int = 1,
        status_report_freq: int = 20
):
    all_labels = read_all_labels(root_folder)
    print(f"Found {len(all_labels)} labels:", all_labels)
    paths_labels = paths_and_labels(root_folder)
    random.Random(shuffle_seed).shuffle(paths_labels)

    total_tfrecords = len(paths_labels) // images_per_tfrecord
    print(f"Number of found images: {len(paths_labels)}")
    print(f"Images per TFRecord: {images_per_tfrecord}")
    print("Total TFRecords to be recorded: ", total_tfrecords)
    count = 0
    it = iter(paths_labels)

    while count <= total_tfrecords:
        tfrecords_name = f"{dataset_name}-{count:05}-{total_tfrecords:05}.tfrecord"
        tfrecords_path = osp.join(tfrecords_folder, tfrecords_name)
        with tf.io.TFRecordWriter(tfrecords_path) as writer:
            print(f"Writing TFRecord {tfrecords_name}")
            for i in range(images_per_tfrecord):
                try:
                    imgpath, raw_label = next(it)
                except StopIteration as e:
                    break
                if i % status_report_freq == 0:
                    print(f"Writing image {i}...")
                image_array, label, imgpath = imarray_label_path(imgpath, image_size, raw_label, all_labels)
                example = encode_example(image_array, label, imgpath)
                serialized_example = serialize_example(example)
                writer.write(serialized_example)
        count += 1


def imarray_label_path(imgpath, size, raw_label, all_labels) -> Tuple[Tensor, int, str]:
    return read_image(imgpath, size), read_label(raw_label, all_labels), imgpath


def encode_example(image_array: Tensor, label: int, imgpath: str) -> tf.train.Example:
    image_shape = image_array.shape
    img_bytes = tf.io.serialize_tensor(image_array)
    return tf.train.Example(
        features=tf.train.Features(feature={
            'image': _bytes_feature(img_bytes),
            'label': _int64_feature(label),
            'height': _int64_feature(image_shape[0]),
            'width': _int64_feature(image_shape[1]),
            'depth': _int64_feature(image_shape[2]),
            'image/filepath': _bytes_feature(imgpath),
        })
    )


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    if isinstance(value, str):
        value = value.encode()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(example: tf.train.Example):
    return example.SerializeToString()


def decode_example(serialized_example):
    feature_description = {
        'image': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.int64),
        'height': tf.io.FixedLenFeature((), tf.int64),
        'width': tf.io.FixedLenFeature((), tf.int64),
        'depth': tf.io.FixedLenFeature((), tf.int64),
        'image/filepath': tf.io.VarLenFeature(tf.string),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)

    image = tf.io.parse_tensor(example['image'], out_type=float)
    image_shape = [example['height'], example['width'], example['depth']]
    image = tf.reshape(image, image_shape)

    return image, example['label'], example['image/filepath']


def read_label(raw_label: str, all_labels: List[str]) -> int:
    all_labels = sorted(all_labels)
    label = all_labels.index(raw_label)
    return label


def read_image(imgpath: str, size: Tuple[int, int]) -> Tensor:
    image = tf.io.read_file(imgpath)
    image = tf.io.decode_image(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, size)
    return image


def read_all_labels(root_folder: str) -> List[str]:
    return [subdir for subdir in os.listdir(root_folder) if osp.isdir(osp.join(root_folder, subdir))]


def paths_and_labels(root_folder) -> List[Tuple[str, str]]:
    ret = list()
    for subdir in os.listdir(root_folder):
        subpath = osp.join(root_folder, subdir)
        if not os.path.isdir(subpath):
            print(f"Found non-folder in root: {subpath}")
            continue
        for img in os.listdir(subpath):
            imgpath = osp.join(subpath, img)
            if not any([imgpath.lower().endswith(ext) for ext in IMAGE_EXTENSIONS]):
                print(f"Found non-image: {imgpath}")
                continue
            raw_label = subdir

            ret.append((imgpath, raw_label))
    return ret