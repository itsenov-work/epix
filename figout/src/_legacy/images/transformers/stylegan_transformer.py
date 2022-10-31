from skimage import transform

from _legacy.images.image_store import imgstore
from _legacy.images.transformers.image_transformer import ImageTransformer, TransformerModes
import numpy as np

class StyleGANTransformer(ImageTransformer):
    def __init__(self, size):
        super(StyleGANTransformer, self).__init__()
        if size & (size - 1) != 0 or size <= 0:
            raise ValueError("Size must be a positive power of two!")
        self.size = size

    def suffix(self):
        return "stylegan" + str(self.size)

    def get_transformer_mode(self):
        return TransformerModes.NP_MODE

    def transform_images(self, folder_name, images, names):
        self._do_transform_images(folder_name, images, names)

    def _do_transform_images(self, folder_name, images, names, num_at_a_time=None):
        '''
        Added a recursive script to handle memory errors
        '''
        transformed_folder = self.get_folder_name(folder_name)
        len_images = images.shape[0]
        if num_at_a_time is None:
            num_at_a_time = len_images
        try:
            start = 0
            stop = num_at_a_time
            while start < len_images:

                if len(images.shape) == 4:
                    dimensions = (num_at_a_time, self.size, self.size, images.shape[3])
                else:
                    dimensions = (num_at_a_time, self.size, self.size)
                image_array = np.zeros(dimensions)
                for i, image in enumerate(images[start:stop]):
                    resized = transform.resize(image, image_array[0].shape)
                    if len(dimensions) == 4:
                        image_array[i, :, :, :] = resized
                    else:
                        image_array[i, :, :] = resized
                imgstore.store_images_as_files_in_path(image_array, "D:\\Projects\\yugiGAN_images", image_names=names)
                del image_array
                start += num_at_a_time
                stop += num_at_a_time
                stop = min(stop, len_images)
        except MemoryError as me:
            next_try = num_at_a_time // 2
            self.log.i("Could not store %d images at once, trying with %d..." % (num_at_a_time, next_try))
            self._do_transform_images(folder_name, images, names, num_at_a_time=next_try)

