import os
import shutil


class GoogleStartup:
    def __init__(self, credentials_filename):
        self.credentials_filename = credentials_filename
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.get_credentials_path()

    def get_credentials_path(self):
        return os.path.join(self.get_credentials_dir(), self.credentials_filename)

    def get_credentials_dir(self):
        return os.path.join("resources", "google", "credentials")

    def move_credentials(self, credentials_local_path):
        if credentials_local_path:
            os.makedirs(self.get_credentials_dir())
            shutil.copy(credentials_local_path, self.get_credentials_path())