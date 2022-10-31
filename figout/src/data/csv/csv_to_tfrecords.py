from typing import Union, List

import pandas
import pandas as pd

from utils.logger import LoggerMixin

IndexList = List[int]
NameList = List[str]
IndicesOrNames = Union[IndexList, NameList]


#TODO
class CSV2TFR(LoggerMixin):

    """
    A class for serializing CSV files into TFRecords format
    """

    def __init__(self,
                 csv_file: str,
                 features: IndicesOrNames,
                 labels: IndicesOrNames,
                 has_headers=False
                 ):
        self.csv_files = csv_file
        self.features = features
        self.labels = labels
        self.has_headers = False

        """
        Type check for features and labels. Both must be of same type, either List[int] or List[str]
        """

        features_type = type(features[0])
        labels_type = type(labels[0])
        allowed_types = [str, int]

        if features_type != labels_type or features_type not in allowed_types:
            self.log.error(f"Features and labels must be either strings [names] or integers [indices], "
                           f"given: {features_type}, {labels_type}")

        """
        Features and labels list must all have same type
        """
        features_or_labels = [(features, features_type, 'features', 'Feature'), (labels, labels_type, 'labels', 'Label')]

        for _list, _type, _text1, _text2 in features_or_labels:
            for idx, f in enumerate(_list):
                if type(f) != _type:
                    self.log.error(f"All selected {_text1} must be of the same type! "
                                   f"{_text2} {idx} has expected type {_type}, actual type {type(f)}")

        headers = pd.read_csv(csv_file, index_col=0, nrows=0).columns.tolist()
