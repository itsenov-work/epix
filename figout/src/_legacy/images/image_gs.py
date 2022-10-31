from utils.google_storage import GoogleStorage
from images.image_store import imgstore
import os


class ImageGS:
    def __init__(self):
        self.gsw = GoogleStorage()

    def upload_pickles(self, folder_name, image_size):
        path = imgstore.get_folder_path(folder_name)
        path = os.path.join(path, "pickledx%d" % image_size)

        self.gsw.upload(path)

    def download_pickles(self, folder_name, image_size):
        path = imgstore.get_folder_path(folder_name)
        path = os.path.join(path, "pickledx%d" % image_size)
        self.gsw.download_to_path(path)
