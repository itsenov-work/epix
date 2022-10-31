from _legacy.images.image_store import imgstore
from utils.logger import LoggerMixin
import os


class TransformerModes:
    NP_MODE = 1
    FILE_MODE = 2


class ImageTransformer(LoggerMixin):
    def make(self, folder_name):
        raw_folder = imgstore.get_folder_path(folder_name)
        transformed_folder = self.get_folder_path(folder_name)
        os.makedirs(transformed_folder, exist_ok=True)

        self.log.i("Making transformed images...")
        tm = self.get_transformer_mode()
        if tm == TransformerModes.FILE_MODE:
            files = os.listdir(raw_folder)
            files = [file for file in files if file.endswith(".png")]
            self.transform_files(folder_name, files)
            self.log.i("Transformed %d files" % len(files))
        elif tm == TransformerModes.NP_MODE:
            images, names = imgstore.provide(folder_name=folder_name, get_image_names=True)
            self.transform_images(folder_name, images, names)
            self.log.i("Transformed %d images" % images.shape[0])

        self.log.end()

    def get(self, folder_name, size=100, get_names=False):
        folder = self.get_folder_name(folder_name)
        self.log.i("Getting transformed images...")
        images = imgstore.provide(folder_name=folder, size_percent=size, get_image_names=get_names)
        self.log.i("Got transformed images!")
        self.log.end()

        return images

    def provide(self, folder_name, size=100, get_names=False):
        self.log.i("Providing transformed images...")
        try:
            return self.get(folder_name, size, get_names=get_names)
        except Exception as e:
            self.log.i("Transformed images are not available.")

            self.make(folder_name)
            return self.get(folder_name, size, get_names=get_names)

    def get_folder_name(self, folder_name):
        return folder_name + "_" + self.suffix()

    def get_folder_path(self, folder_name):
        folder_name = self.get_folder_name(folder_name)
        return imgstore.get_folder_path(folder_name)

    '''
    These are the methods you need to implement in order to inherit this class
    '''
    def suffix(self):
        raise NotImplementedError

    def get_transformer_mode(self):
        raise NotImplementedError

    def transform_files(self, folder_name, files):
        pass

    def transform_images(self, folder_name, images, names):
        pass