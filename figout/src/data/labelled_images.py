"""drones_poles_metal dataset."""

import tensorflow_datasets as tfds
import cv2
from tensorflow_datasets.core import PathLike

from utils.google_storage import GoogleStorage
from utils.yaml_config import YamlConfigBuilder


class LabelledImageDataset(YamlConfigBuilder, tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for a labelled image dataset."""

    REQUIRED_FIELDS = ["name", "num_classes", "image_dims"]
    OPTIONAL_FIELDS = {
        # TODO
        "description": "Test description"
    }

    def __init__(self, filename: PathLike):
        self.load_yaml(filename)
        self.__class__.VERSION = tfds.core.Version('1.0.0')
        self.__class__.MANUAL_DOWNLOAD_INSTRUCTIONS = "Blah"
        self.name = self.getprop("name")
        super(LabelledImageDataset, self).__init__()

    """---------Public API functions---------"""

    def generate(self, path):
        """
        Generates tf.Example dataset from local folder.
        The folder should have the following structure:

        'train':
            -> 'label1'
                -> 1.jpg 2.jpg 3.jpg
            -> 'label2'
                -> 1.png 2.png 3.png
            ...
        """
        download_config = tfds.download.DownloadConfig(manual_dir=path)
        self.download_and_prepare(download_config=download_config)

    def upload(self, bucket):
        GoogleStorage(bucket).upload(self.data_dir, self.gs_path)

    def download(self, bucket, local_path):
        GoogleStorage(bucket).download_to_path(self.gs_path, local_path)

    def load(self, local_path):
        return tfds.load(self.name, data_dir=local_path)

    @property
    def gs_path(self):
        return f'datasets/{self.name}'

    """-------------Private functions-------------"""

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=self.getprop("description"),
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=self.getprop("image_dims")),
                    "label": tfds.features.ClassLabel(num_classes=self.getprop("num_classes")),
                    "image/filename": tfds.features.Text(),
                }
            ),
            supervised_keys=("image", "label"),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        path = dl_manager.manual_dir
        return {
            'train': self._generate_examples(path)
        }

    def _generate_examples(self, path):
        builder = tfds.ImageFolder(path)
        dataset = builder.as_dataset(split='train')
        i = 0
        for data in dataset:
            i += 1
            image = data['image']
            image = cv2.resize(image.numpy(), self.getprop("image_dims")[:2])
            example = {
                key: value.numpy() for key, value in data.items() if key != "image"
            }
            example['image'] = image
            yield i, example


