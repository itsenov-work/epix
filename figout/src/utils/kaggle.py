import contextlib
import os
import json
from pathlib import Path

from utils.dir import Dir


class Kaggle:

    def __init__(self):
        resources = Dir.get_resources_dir()
        kaggle_folder = os.path.join(resources, 'kaggle')
        os.environ["KAGGLE_CONFIG_DIR"] = kaggle_folder
        os.chmod(os.path.join(kaggle_folder, 'kaggle.json'), 777)

        import kaggle
        self.api = kaggle.KaggleApi()
        self.kaggle_folder = kaggle_folder
        self.cache_file = os.path.join(self.kaggle_folder, 'cache.json')
        os.makedirs(self.kaggle_folder, exist_ok=True)

        if not os.path.exists(self.cache_file):
            self.write_cache({})

        self.download_folder = os.path.join(self.kaggle_folder, 'downloads')
        self.api.authenticate()

    def download_dataset(self, url, path=None, force=False):
        dataset_url = self.get_dataset_url(url)
        folder_name = self.get_folder_name(url)
        self.api.dataset_download_files(dataset_url)
        if path is None:
            path = self.get_download_path(folder_name)

        if not force:
            cached = self.check_cache(dataset_url)
            if cached is not None:
                return cached

        os.makedirs(path, exist_ok=True)
        self.do_download(dataset_url, path, force)
        return path

    def do_download(self, dataset_url, path, force):
        self.api.dataset_download_files(dataset=dataset_url, path=path, unzip=True, force=force, quiet=False)
        self.update_cache(dataset_url, path)

    def get_download_path(self, folder_name):
        return os.path.join(self.download_folder, folder_name)

    def get_folder_name(self, dataset_url):
        kaggle_url = self.get_dataset_url(dataset_url)
        return kaggle_url.split("/")[-1]

    def get_dataset_url(self, dataset_url):
        return dataset_url.split("kaggle.com/")[-1]

    def check_cache(self, url):
        cache = self.read_cache()
        return cache[url] if url in cache else None

    def read_cache(self):
        with open(self.cache_file, 'r') as f:
            return json.load(f)

    def write_cache(self, contents):
        with open(self.cache_file, 'w') as f:
            return json.dump(contents, f)

    def update_cache(self, key, value):
        cache = self.read_cache()
        cache[key] = value
        self.write_cache(cache)