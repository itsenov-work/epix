from _legacy.images.transformers.image_transformer import ImageTransformer, TransformerModes
from _legacy.images.image_utilities import getLaplacian
from _legacy.images.image_store import imgstore
import numpy as np


class LaplacianTransformer(ImageTransformer):
    def suffix(self):
        return "edges"

    def get_transformer_mode(self):
        return TransformerModes.NP_MODE

    def transform_images(self, folder_name, images, names):
        edge_images = getLaplacian(images)
        edge_images = edge_images / np.max(np.abs(edge_images))
        transformed_folder = self.get_folder_name(folder_name)
        imgstore.store_images_as_files(edge_images, folder_name=transformed_folder, image_names=names)
