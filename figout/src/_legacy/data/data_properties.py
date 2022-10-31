from abc import ABC
from enum import Enum, auto
from typing import Tuple


class DataType(Enum):
    IMAGE = auto()

    def folder_name(self):
        return self.name.lower()


class DimensionMixin(ABC):
    def get_dimensions(self):
        raise NotImplementedError


class SizeMixin:
    size = None

    def set_size(self, size):
        self.size = size


class DimensionProvider(DimensionMixin):
    def __init__(self, dimensions: Tuple):
        self.dimensions = dimensions

    def get_dimensions(self):
        return self.dimensions
