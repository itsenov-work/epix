import json
import os.path as osp
from typing import List, Dict

from utils.csv_utils import read_csv_file
from utils.filesystem_utils import search_path_recursive
from utils.logging import Logger, my_tqdm
import pandas as pd

logger = Logger('SampleReader')


class EPIXMethylationDatabase:

    def __init__(self, local_directory):
        self.directory = local_directory
        self.metadata = self._get_all_metadata()

    def get_sample(self, sample_id, cpgs: List[str] = None) -> List:
        """
        Get the CPG beta values for a single data sample.

        :param sample_id: The unique identifier for a sample, ex. GSM3039543
        :param cpgs: The cpg names which you want to get the values for,
                     ex. ['cg3925153', 'cg57739823'].
                     if None (default), returns the beta values for all cpgs.
        :return: A list.
                 For ease of use, the first item is the sample id.
                 Followed by the cpg beta values of that sample.

        Note: This returns ONLY the CPG beta values.
        Metadata for the sample and the cpg names must be queried separately.

        !!IMPORTANT!! if the cpgs given do not match the available cpgs for the sample,
        the function will throw an error.

        It is recommended that the cpgs are filtered by availability before using
        this function.

        Example use: get_sample('GSM3599934', ['cg343482']) -> ['GSM3599934', 0.0335]
        where 0.0335 is the beta value for cg343482 on sample GSM3599934.

        """
        return self.get_samples([sample_id], cpgs)[0]

    def get_samples(self, sample_ids: List[str], cpgs: List[str] = None) -> List[List]:
        """
        Get the CPG beta values for multiple data samples.
        :return: list of samples, each a list of cpg beta values.
        See get_sample()
        """

        return self._do_get_samples(sample_ids, cpgs)

    def get_sample_metadata(self, sample_id: str) -> Dict:
        """
        Get the metadata (anything that is not CPG beta values) for a single data sample.

        :param sample_id: The unique identifier for a sample, ex. GSM3039543
        :return: The metadata as a dictionary of key-value pairs, ex. 'disease': 'schizophrenia'
        """
        return self.metadata[sample_id]

    def get_sample_cpgs(self, sample_id: str) -> List[str]:
        """
        Get the list of CPG names for a single sample.
        Note: This is different from get_sample, which returns the beta values
        for those CPGs.

        :param sample_id: The unique identifier for a sample, ex. GSM3039543
        :return: The cpgs as a list of strings.
        """
        folder = self._get_sample_folder(sample_id)
        return self._get_folder_cpgs(folder)

    def get_common_cpg(self, sample_ids: List[str]) -> List[str]:
        """
        Get the list of CPGs that are available for all given sample ids.

        :param sample_ids: List of unique sample identifiers.
        :return: List of CPGs you can query all given samples for.
                 The list is sorted alphabetically.
        """
        return self._do_get_common_cpg(sample_ids)

    def search(self, key_values: dict):
        """
        Filter samples based on key - value pair criteria.

        :param key_values: A dictionary of filter criteria

        :return:
        """
        return self._do_search(key_values)


    @staticmethod
    def pick_random_samples(sample_ids, n, seed=7):
        import random
        random.seed(seed)
        return random.sample(sample_ids, n)

    def get_data(self, config):
        all_sample_ids = []
        for request in my_tqdm(config):
            key_values = request['filters']
            sample_ids = self.search(key_values)
            if 'n' in request:
                n = request['n']
                if 'random' in request and request['random']:
                    if 'seed' in request:
                        sample_ids = self.pick_random_samples(sample_ids, n, request['seed'])
                    else:
                        sample_ids = self.pick_random_samples(sample_ids, n)
                else:
                    sample_ids = sample_ids[:n]
            all_sample_ids.extend(sample_ids)

        logger.i(f"Total samples: {len(all_sample_ids)}")
        common_cpgs = self.get_common_cpg(sample_ids=all_sample_ids)
        samples = self.get_samples(sample_ids=all_sample_ids, cpgs=common_cpgs)
        metadata_ret = [{**self.get_sample_metadata(sample_id), 'sample_id': sample_id} for sample_id in all_sample_ids]

        df = pd.DataFrame(samples, columns=['sample_id'] + common_cpgs)
        return df, metadata_ret

    ###############################################################################
    #                             PRIVATE METHODS                                 #
    ###############################################################################
    def _get_sample_folder(self, sample_id):
        return self.get_sample_metadata(sample_id)['folder']

    def _get_all_metadata(self, override=True):
        all_metadata_file = osp.join(self.directory, 'all_metadata.json')

        if osp.exists(all_metadata_file) and override is not True:
            with open(all_metadata_file, 'r') as f:
                return json.load(f)

        metadata_files = search_path_recursive(self.directory, 'metadata.json')

        all_metadata = {}

        for file in metadata_files:
            with open(file, 'r') as f:
                metadata = json.load(f)
                folder = osp.dirname(file)
                for sample_id in metadata:
                    metadata[sample_id]['folder'] = folder
                all_metadata.update(metadata)

        with open(all_metadata_file, 'w+') as f:
            json.dump(all_metadata, f)

        return all_metadata

    def _read_sample_csv(self, sample_id):
        folder = self._get_sample_folder(sample_id)
        sample_file = f'{sample_id}.csv'
        with open(osp.join(folder, sample_file), 'r') as f:
            return read_csv_file(f)

    @staticmethod
    def _get_folder_cpgs(folder):
        cg_list = 'cg_list.csv'
        with open(osp.join(folder, cg_list), 'r') as f:
            return read_csv_file(f)

    def _parse_sample(self, sample_id: str, cpgs: List[str] = None):
        """
        Parse a single sample. The folder (locally) where the sample resides must be given.
        If cpgs is None, the whole sample will be read.
        Otherwise, only the cpgs given in the cpgs list will be read in the order they are given.
        """

        logger.i(f"Parsing sample {sample_id}...")
        sample = self._read_sample_csv(sample_id)
        sample_cpgs = self.get_sample_cpgs(sample_id)

        index_dict = {k: i for i, k in enumerate(sample_cpgs)}

        if cpgs is not None:
            indices = [index_dict[k] for k in cpgs]
            if len(indices) < len(cpgs):
                logger.e(f"The cpg list contains cpgs that are not present in the sample. "
                         f"Make sure to filter the cpgs before reading the sample.")
                logger.error(f"Error while parsing sample {sample_id}.")
            sample = [sample[i] for i in indices]

        return [sample_id] + sample

    def _do_get_samples(self, sample_ids: List[str], cpgs: List[str] = None):
        ret = []
        logger.start(f"Parsing {len(sample_ids)} samples:")
        for sample in my_tqdm(sample_ids):
            ret.append(self._parse_sample(sample, cpgs))
        logger.s(f"{len(sample_ids)} samples parsed successfully.")
        logger.end()
        return ret

    def _do_get_common_cpg(self, sample_ids):
        logger.start(f"Getting common cpg from {len(sample_ids)} samples.")
        common_cpg = None
        folders = set([self._get_sample_folder(sample_id) for sample_id in sample_ids])
        logger.i(f"Looking in {len(folders)} folders...")
        for folder in my_tqdm(folders):
            sample_cpgs = self._get_folder_cpgs(folder)

            if common_cpg is None:
                common_cpg = set(sample_cpgs)

            common_cpg = common_cpg.intersection(set(sample_cpgs))
        logger.s(f"Successfully got all {len(common_cpg)} common cpgs for {len(sample_ids)} samples.")
        return sorted(list(common_cpg))

    def _do_search(self, key_values: dict):
        def key_value_correct(key, value, sample):
            return key in sample and sample[key] == value

        samples = [k for k in self.metadata if
                   all([key_value_correct(key, key_values[key], self.metadata[k]) for key in key_values])]
        logger.i(f"Found {len(samples)} fulfilling criteria {key_values}")
        return sorted(samples)


if __name__ == '__main__':
    local_path = r'C:\Users\itsen\workspaces\epix\epigenetic data\diseases\Samplified Diseases'

    config = [
        {
            "filters": {
                "disease": "type 2 diabetes",
                "tissue": "whole blood",
                "sample_type": "case"
            },
        },
        {
            "filters": {
                "disease": "schizophrenia",
                "tissue": "whole blood",
                "sample_type": "case"
            },
            "n": 10,
            "random": True,
            "seed": 10
        },
        {
            "filters": {
                "disease": "type 2 diabetes",
                "tissue": "whole blood",
                "sample_type": "control"
            },
            "n": 40,
            "random": False,
        },
    ]
    EPIXMethylationDatabase(local_directory=local_path).get_data(config)