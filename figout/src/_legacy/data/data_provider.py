from threading import RLock
import abc

import tensorflow as tf
from _legacy.data.data_reader import FolderDataReader
from _legacy.data.data_properties import SizeMixin, DimensionMixin
from _legacy.images.standard_datasets import StandardDatasets
from utils.lock import synchronized_object
from utils.decorators import MRODecoratedMixin, MRODecorator


class DataProvider(DimensionMixin, MRODecoratedMixin, abc.ABC):
    def __init__(self):
        self.lock = RLock()
        self.load()

    @MRODecorator(synchronized_object(lock_name="lock"))
    def load(self):
        raise NotImplementedError

    @MRODecorator(synchronized_object(lock_name="lock"))
    def get(self, n):
        raise NotImplementedError

    def get_batch(self):
        raise NotImplementedError


class SizeDataProviderMixin(SizeMixin):
    pass


class WrappedDataProvider(DataProvider):

    def __init__(self):
        self.dataset = None
        self.iterator = None
        super(WrappedDataProvider, self).__init__()

    def load(self):
        del self.dataset
        self.dataset = self._prepare_dataset()
        self.iterator = iter(self.dataset)
        return self

    def get_batch(self):
        return next(self.iterator)

    def get(self, n):

        ret = self.get_batch()
        n = n - ret.shape[0]

        while n > 0:
            batch = self.get_batch()
            n = n - ret.shape[0]
            ret = tf.concat((ret, batch), 0)

        if n < 0:
            ret = ret[:n, ...]
        return ret

    def get_dimensions(self):
        return self.get(1)[0].shape

    def _prepare_dataset(self) -> tf.data.Dataset:
        raise NotImplementedError


class FolderProvider(WrappedDataProvider):
    def __init__(self, data_folder: str, batch_size: int, data_reader: FolderDataReader):
        self.data_folder = data_folder
        self.data_reader = data_reader
        self.batch_size = batch_size
        super(FolderProvider, self).__init__()

    def _prepare_dataset(self):
        dataset = self.data_reader.get(self.data_folder)
        return dataset.batch(self.batch_size)


class StandardTFDSProvider(WrappedDataProvider, SizeDataProviderMixin):
    def __init__(self, batch_size: int, dataset_name: str, split: str = None, labelled: bool = None):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.dataset_provider = StandardDatasets()
        self.dataset_info = None
        if split is not None:
            self.split = split
        else:
            self.split = 'train'
        if labelled is not None:
            self.labelled = labelled
        else:
            self.labelled = False
        super(StandardTFDSProvider, self).__init__()

    def _prepare_dataset(self):
        data = self.dataset_provider.provide(self.dataset_name, self.batch_size, self.labelled, self.split)
        self.dataset_info = self.dataset_provider.dataset_info
        return data
