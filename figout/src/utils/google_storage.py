import os

from google.cloud import storage
from utils.logger import LoggerMixin


class GoogleStorage(LoggerMixin):
    downloads_folder = os.path.join("resources", "google", "downloads")

    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        super(GoogleStorage, self).__init__()

    def download(self, source_path, destination_name=None, force=True, tar=True):
        source_name = source_path.split("/")[-1]

        if destination_name is None:
            destination_name = source_name

        self.log.start("Downloading from Google Storage, bucket {}".format(self.bucket_name))
        destination_path = self.get_destination_path(destination_name)

        if tar:
            if not source_path.endswith(".tar.gz"):
                source_path += ".tar.gz"
            if not destination_path.endswith(".tar.gz"):
                destination_path += ".tar.gz"

        if os.path.exists(destination_path):
            if force:
                os.remove(destination_path)
            else:
                self.log.ok("The destination path already exists. If you wish to override, re-run with force=True")
                self.log.end()
                return destination_path
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        self._download_blob(source_path, destination_path)
        self.log.end()

        return destination_path

    def download_to_path(self, source_path, destination_path=None, tar=True, force=True):
        self.log.start("Downloading from Google Storage, bucket {}".format(self.bucket_name))
        if destination_path is None:
            destination_path = source_path

        if tar:
            source_tar_path = source_path + ".tar.gz"
            destination_tar_path = destination_path + ".tar.gz"
            destination_folder_path = os.path.join(*(destination_tar_path.split(os.path.sep)[:-1]))
            os.makedirs(destination_folder_path, exist_ok=True)
            self._download_blob(source_tar_path, destination_tar_path)
            import tarfile

            tar = tarfile.open(destination_tar_path, "r:gz")
            tar.extractall(path=os.path.dirname(destination_tar_path))
            tar.close()
            self.log.i("Files extracted successfully.")
            os.remove(destination_tar_path)
        else:
            destination_folder_path = os.path.join(*(destination_path.split(os.path.sep)[:-1]))
            os.makedirs(destination_folder_path, exist_ok=True)
            self._download_blob(source_path, destination_path)

        self.log.end()

        return destination_path

    def upload(self, source_path, destination_path=None):
        self.log.start("Attempting to upload to Google Storage, bucket {}".format(self.bucket_name))

        if destination_path is None:
            destination_path = source_path

        self._upload_blob(source_path, destination_path)
        self.log.end()

        return destination_path

    def get_destination_path(self, destination_name):
        return os.path.join(self.downloads_folder, destination_name)

    def _download_blob(self, source_blob_name, destination_file_name):
        """Downloads a blob from the bucket."""

        storage_client = storage.Client()
        source_blob_name = "/".join(source_blob_name.split(os.path.sep))
        self.log.i("Source path: {}".format(source_blob_name))
        self.log.i("Destination path: {}".format(destination_file_name))

        bucket = storage_client.bucket(self.bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)

        self.log.i("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))

    def _upload_blob(self, source_path, destination_path):
        """Uploads a blob to the bucket."""
        storage_client = storage.Client()

        bucket = storage_client.bucket(self.bucket_name)
        destination_path = "/".join(destination_path.split(os.path.sep))

        if os.path.isdir(source_path):
            import tarfile
            blob = bucket.blob(destination_path + ".tar.gz")
            tar_path = source_path + ".tar.gz"
            self.log.i("Source path: {}".format(tar_path))
            self.log.i("Destination path: {}".format(destination_path + ".tar.gz"))

            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(source_path, arcname=os.path.basename(source_path))

            blob.upload_from_filename(tar_path)
            os.remove(tar_path)
        else:
            blob = bucket.blob(destination_path)
            self.log.i("Source path: {}".format(source_path))
            self.log.i("Destination path: {}".format(destination_path))
            blob.upload_from_filename(source_path)

        self.log.i("Blob {} uploaded to {}.".format(source_path, destination_path))
