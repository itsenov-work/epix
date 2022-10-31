import pandas as pd
import os.path as osp

from utils.logger import LoggerMixin


#
# 1. Take a disease dataset - select [] columns from it
# 2. Take multiple diseases - find common cpg, merge datasets with extra column depicting dataset name
# 3. Formatting and persistence
#       - Serialize each run into tfrecords
#       - Save each run to gcloud
#



class CSVCreator(LoggerMixin):

    def __init__(self, root_dir: None):
        super(CSVCreator, self).__init__()
        if root_dir is None:
            root_dir = osp.join('resources', 'csv')
            root_dir = osp.abspath(root_dir)
        if not osp.exists(root_dir):
            self.root_dir = root_dir
        self.log.i(f"Initialized CSVCreator in path {self.root_dir}")


