import os
from abc import ABC
from utils.google_storage import GoogleStorage as GS
from utils.kaggle import Kaggle


class DataDownloader(ABC):
    download_path = None

    def __init__(self):
        self.download()

    def download(self):
        raise NotImplementedError


class GoogleStorageDownloader(DataDownloader):
    def __init__(self, bucket_name, google_path, download_path=None, tar=True):
        self.google_path = google_path
        self.download_path = download_path
        self.tar = tar
        super(GoogleStorageDownloader, self).__init__()

    def download(self):
        self.download_path = GS().download_to_path(self.google_path, self.download_path, tar=self.tar)


class KaggleDownloader(DataDownloader):
    def __init__(self, kaggle_url, download_path=None, force=False):
        self.kaggle_url = kaggle_url
        self.download_path = download_path
        self.force = force
        super(KaggleDownloader, self).__init__()

    def download(self):
        self.download_path = Kaggle().download_dataset(self.kaggle_url, self.download_path, self.force)
