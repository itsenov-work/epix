import os
import shutil

from utils.logger import LoggerMixin


class MiscUtils(LoggerMixin):
    def clear_folder(self, folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                self.log.e('Failed to delete %s. Reason: %s' % (file_path, e))