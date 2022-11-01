import datetime
import time
import typing
from collections import OrderedDict

import requests
import json
from strenum import StrEnum
import urllib.parse

GLOBAL_URL = "https://ngdc.cncb.ac.cn/ewas/"


class BrowseEndpoints(StrEnum):
    ASSOCIATION = "association"
    GENE = "gene"
    TRAIT = "trait"
    PUBLICATION = "publication"
    STUDY = "study"


class BrowseFilters(StrEnum):
    __order__ = "TRAIT TRAIT2 ONTOLOGY TISSUE PMID STUDY_ID GENE CPG_INDEX CHROMOSOME POS_START POS_END " \
                "RANK_START RANK_END CPG_ISLAND YEAR TYPE FEATURE ORDER OFFSET LIMIT"
    TRAIT = "traitList"
    TRAIT2 = "!traitList"
    ONTOLOGY = "ontology"
    TISSUE = "tissues"
    PMID = "pmid"
    STUDY_ID = "studyId"
    GENE = "gene"
    CPG_INDEX = "probeId"
    CHROMOSOME = "chr"
    POS_START = "posStart"
    POS_END = "posEnd"
    RANK_START = "rankStart"
    RANK_END = "rankEnd"
    CPG_ISLAND = "cpg"
    YEAR = "year"
    TYPE = "type"
    FEATURE = "feature"
    ORDER = "order"
    OFFSET = "offset"
    LIMIT = "limit"

    @staticmethod
    def get_defaults() -> typing.OrderedDict:
        defaults = OrderedDict()
        for t in BrowseFilters:
            defaults[t] = ""
        defaults['limit'] = 10
        defaults['order'] = 'asc'
        defaults['offset'] = 0
        return defaults

    @classmethod
    def get_endpoint(cls, filter):
        if filter == BrowseFilters.TRAIT:
            return "phenotype"
        if filter == BrowseFilters.CPG_INDEX:
            return "probe_id"
        if filter == BrowseFilters.TISSUE:
            return "tissue"
        if filter == BrowseFilters.ONTOLOGY:
            return "ontology"
        if filter == BrowseFilters.PMID:
            return "pmid"
        if filter == BrowseFilters.STUDY_ID:
            return "study_id"


def get_all_filters(filter_type):
    endpoint = BrowseFilters.get_endpoint(filter_type)
    timestamp = int(datetime.datetime.now().timestamp() * 1000)
    all_filters = {}
    url = f"https://ngdc.cncb.ac.cn/ewas/{endpoint}?term=_&_={timestamp}"
    r = requests.get(url)
    for result in r.json()['results']:
        if result['id'] in all_filters:
            continue
        all_filters[result['id']] = result['text']
    save_json("ewas_database", f"all_{filter_type}", all_filters)
    return all_filters


def _get_browse_url(keyword, filters):
    default_filters = BrowseFilters.get_defaults()
    default_filters.update(filters)
    if BrowseFilters.TRAIT not in filters:
        default_filters.pop(BrowseFilters.TRAIT2)
    url = GLOBAL_URL + f"browse/{keyword}?"
    url += "&".join([f"{key}={value}" for key, value in default_filters.items()])
    url = url.replace("!traitList", "traitList")
    return url


def browse(endpoint, filters):
    url_filters = {}
    for key, value in filters.items():
        url_filters[key] = urllib.parse.quote(str(value))
    url = _get_browse_url(endpoint, url_filters)
    print(url)
    r = requests.get(url)
    return r.json()


def save_json(folder, filename, obj):
    import os
    import os.path as osp
    folder_path = osp.join('..', 'resources', folder)

    os.makedirs(folder_path, exist_ok=True)
    filepath = osp.join(folder_path, filename + ".json")
    with open(filepath, 'w+') as f:
        json.dump(obj, f, sort_keys=True, indent=4)


def get_json(folder, filename):
    import os
    import os.path as osp
    folder_path = osp.join('..', 'resources', folder)

    os.makedirs(folder_path, exist_ok=True)
    filepath = osp.join(folder_path, filename + ".json")
    with open(filepath, 'r') as f:
        return json.load(f)


def iterate_browse(endpoint, filters_list):
    i = 0
    while i < len(filters_list):
        filters = filters_list[i]
        try:
            results = browse(endpoint, filters)
            time.sleep(1)
            i += 1
            yield filters, results

        except Exception as e:
            print(f"Retrying filter {filters} in 10 seconds...")
            time.sleep(10)


def get_all_blood_traits():
    tissues = get_json("ewas_database", "all_tissues")
    blood_tissues = [tissue for tissue in tissues if "blood" in tissue]
    print(blood_tissues)

    blood_traits = {}

    filters_list = [{
        BrowseFilters.TISSUE: tissue,
        BrowseFilters.LIMIT: 10
    } for tissue in blood_tissues]

    for filters, result in iterate_browse(BrowseEndpoints.TRAIT, filters_list=filters_list):
        print(f"Total results for tissue {filters[BrowseFilters.TISSUE]}: {result['total']}")
        result = result['content']
        for r in result:
            trait = r['trait']
            if trait not in blood_traits:
                blood_traits[trait] = list()
                print(len(blood_traits))
            blood_traits[trait].append(filters[BrowseFilters.TISSUE])

    save_json("ewas_database", "blood_traits_2nd_method", blood_traits)


if __name__ == '__main__':
    get_all_blood_traits()
