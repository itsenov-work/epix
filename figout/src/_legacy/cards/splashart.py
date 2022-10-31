import os
from PIL import Image
from images.image_store import imgstore
from images.transformers.image_transformer import ImageTransformer, TransformerModes


class SplashArtTransformer(ImageTransformer):
    folder = "splash_art"

    def transform_files(self, folder_name, files):
        raw_folder = imgstore.get_folder_path(folder_name)
        transformed_fodler = self.get_folder_path(folder_name)
        for idx, file in enumerate(files):
            path = os.path.join(raw_folder, file)
            image = Image.open(path)
            width, height = image.size
            left = int(38 / 322 * width)
            top = int(78 / 433 * height)
            right = int(284 / 322 * width)
            bottom = int(305 / 433 * height)
            splash = image.crop((left, top, right, bottom))
            splash.save(os.path.join(transformed_fodler, file))
            if (idx + 1) % 100 == 0:
                self.log.i("Created splash art for %d cards." % (idx + 1))

    def suffix(self):
        return "splash_art"

    def get_transformer_mode(self):
        return TransformerModes.FILE_MODE
