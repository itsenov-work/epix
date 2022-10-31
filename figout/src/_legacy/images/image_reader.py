from _legacy.data.data_reader import FolderDataReader
from _legacy.images.transformers.image_transformer import ImageTransformer


class FolderImageReader(FolderDataReader):
    def __init__(self, transformer: ImageTransformer, size=100):
        self.transformer = transformer
        self.size = size

    def get(self, folder):
        return self.transformer.provide(folder, size=self.size)
