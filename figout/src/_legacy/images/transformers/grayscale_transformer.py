from _legacy.images.transformers.image_transformer import ImageTransformer, TransformerModes
from _legacy.images.image_utilities import toGrayscale
from _legacy.images.image_store import imgstore


class GrayscaleTransformer(ImageTransformer):
    def suffix(self):
        return "grayscale"

    def get_transformer_mode(self):
        return TransformerModes.NP_MODE

    def transform_images(self, folder_name, images, names):
        gs_images = toGrayscale(images)
        transformed_folder = self.get_folder_name(folder_name)
        imgstore.store_images_as_files(gs_images, folder_name=transformed_folder, image_names=names)
