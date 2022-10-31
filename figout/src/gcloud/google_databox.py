import os

from utils.dir import Dir
from utils.logger import LoggerMixin


class GoogleDataBox(LoggerMixin):
    """
    A class to handle data from GoogleStorage.
    This works with a yaml of the following structure:
    {
        'credentials' [required]: name of the GCloud credentials file, found in resources/google/credentials
        'bucket' [required]: name of the GS bucket
        'path' [required]: the upstream in-bucket path to the resource
        'force' [bool, required]: whether to re-download the data forcefully (delete local .tar file)
        'force_extraction' [bool, optional]: whether to re-extract the data forcefully (delete local folder)
        'download_folder' [optional]: the local download path,
        'extraction_folder' [optional]: the extraction path. NOTE: all folder uploaded in GCloud should be tarred.
        'subdirs' [List[str], optional]: where to find the data itself within the extraction folder
    }
    """

    credentials = None
    bucket = None
    path = None
    extraction_folder = None
    force = None
    force_extraction = None
    download_folder = None
    # the path after download is complete
    downloaded_path = None
    # whether the box has been invoked
    prepared = False
    subdirs = None
    # the local path where you can find the data after preparation
    local_path = None

    def __init__(self, filename):
        super(GoogleDataBox, self).__init__()
        import yaml
        self.filename = filename
        if not filename.endswith(".yaml"):
            filename = filename + ".yaml"

        file_path = os.path.join(Dir.get_resources_dir(), "data_configs", filename)
        with open(file_path) as f:
            self.yaml = yaml.load(f, Loader=yaml.Loader)

        # Add json fields to class instance
        for x in ("extraction_folder", "download_folder", "path", "bucket",
                  "credentials", "force", "force_extraction", "subdirs"):
            setattr(self, x, self.yaml.get(x))

        # Check required fields
        for x in ("path", "bucket", "credentials", "force"):
            if getattr(self, x) is None:
                self.log.error("Json file {} is missing required field {}".format(file_path, x))

        self.name = self.path.split("/")[-1]

    def prepare(self):
        self.log.start(f"Preparing data config {self.filename}...")
        self.initialize_google_cloud()
        self.download()
        self.extract()
        self.report_success()

    def initialize_google_cloud(self):
        from utils.google_startup import GoogleStartup
        GoogleStartup(self.credentials)
        self.log.ok("GCloud initialized")

    def download(self):
        from utils.google_storage import GoogleStorage
        gs = GoogleStorage(self.bucket)
        self.downloaded_path = gs.download(self.path, force=self.force)

    def extract(self):
        import tarfile
        if self.extraction_folder is None:
            self.extraction_folder = os.path.join(Dir.get_resources_dir(), "extracted_data", self.name)
        self.local_path = os.path.join(self.extraction_folder, *self.subdirs)

        if os.path.exists(self.extraction_folder):
            if not self.force_extraction:
                self.log.ok("The extraction directory exists. "
                            "Assuming that the folder has been extracted before successfully. Will not extract again.")
                return
            import shutil
            self.log.w("Extraction directory exists. It will be overridden.")
            shutil.rmtree(self.extraction_folder)

        os.makedirs(self.extraction_folder, exist_ok=False)
        tar = tarfile.open(self.downloaded_path, "r:gz")
        tar.extractall(path=self.extraction_folder)
        tar.close()

    def report_success(self):
        self.log.s(f"Data {self.name} prepared!")
        self.log.end()