import GEOparse
import urllib.request
import pickle
import os
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from operator import itemgetter
import ast
from typing import List, Dict

DATASET_FILES = r"F:\data\epigenetics"


def name_from_code(gse_or_illness):
    dict_of_names = {
        # TRAIN:  ~1465 samples
        'GSE106648': "multiple_sclerosis",  # lots of NaNs
        'GSE125105': "depression",  # manual
        'GSE19711': "ovarian_cancer",  # everything has NaNs
        'GSE27044': "autism",
        'GSE30870': "newborns",  # all have NaN
        'GSE40279': "ethnicity",  # perfect
        'GSE52588': "down",  # all have NaNs
        'GSE41037': "schizophrenia",  # all have NaNs
        'GSE53740': "neurodegen",  # perfect
        # 'GSE58119': "breast_cancer",  # bug still
        'GSE67530': "ARDS",  # all have NaNs
        'GSE77445': "childhood_trauma",  # perfect
        'GSE77696': "HIV",  # manual
        'GSE81961': "crohn_disease",  # perfect
        'GSE84624': "kawasaki_disease",  # all have NaNs
        'GSE97362': "charge_and_kabuki",  # all have NaNs
        # TEST:  Data for 281 patients out of 420 was written.
        'GSE102177': "siblings_maternial_diabetes",  # perfect
        'GSE103911': "cSCC",  # perfect
        'GSE105123': "acclim_to_high_altitude",  # perfect
        'GSE107459': "pregnancy",  # all have NaNs
        'GSE107737': "congenital_hypopituitarism",  # perfect
        'GSE112696': "myasthenia_gravis",  # perfect
        'GSE34639':  "food_allergy",  # perfect
        'GSE37008': "early-life_experiences",  # all have NaNs
        # # 'GSE59065': "immunocompetence",  # manual
        # # 'GSE61496': "birth-weight_discordant_twins",  # manual
        'GSE79329': "persistent_organic_pollutants",  # perfect
        # # 'GSE87582': "hiv_cognitive_impairement",  # not applicable
        'GSE87640': "IBD",  # perfect
        'GSE98876': "under_alcohol_treatment",  # perfect
        'GSE99624': "osteoporotic"  # all have NaNs

    }
    if gse_or_illness == 'all':
        return list(dict_of_names.keys())
    if gse_or_illness in dict_of_names.keys():
        return dict_of_names[gse_or_illness]
    else:
        for gse, illness in dict_of_names.items():
            if gse_or_illness == illness:
                return gse


def _load_from_url(url):
    url = manual_link_fixes(url)
    with urllib.request.urlopen(url) as r:
        data = r.read()
    return data


def manual_link_fixes(url):
    if ',' in url:
        url = [link for link in url.split(',')]
        return url[1]

    return url


def _load_from_pkl(data_dir):
    with open(data_dir, 'rb') as d:
        return pickle.load(d)


def load_dataset(dataset_name, healthy=True, case=True, specific_fields=None):
    print("Loading {}".format(dataset_name))
    data_dir = os.path.join(DATASET_FILES, dataset_name)
    data = list()
    for data_file in os.listdir(data_dir):
        if os.path.isfile(os.path.join(data_dir, data_file)):
            if "complete" in data_file:
                if "depression" in data_file.lower():
                    cpg_for_manual = ['age', 'is_male', 'is_case'] + specific_fields
                    data = pd.read_csv(os.path.join(data_dir, data_file), usecols=cpg_for_manual)
                elif "hiv" in data_file.lower():
                    cpg_for_manual = ['age', 'sex', 'case'] + specific_fields
                    data = pd.read_csv(os.path.join(data_dir, data_file), usecols=cpg_for_manual)
                    data = data.rename(columns={'sex': "is_male", 'case': "is_case"})
            if healthy and case:
                data = data + _load_from_pkl(os.path.join(data_dir, data_file))
            elif healthy and not case:
                if "control" in data_file:
                    data = data + _load_from_pkl(os.path.join(data_dir, data_file))
            elif case and not healthy:
                if "case" in data_file:
                    data = data + _load_from_pkl(os.path.join(data_dir, data_file))
            else:
                print("Cant have neither healthy nor sick people.")
    print("Data for {} loaded".format(dataset_name))
    return data


def load_only_one(dataset_name):
    print("Collecting " + dataset_name)
    data_dir = os.path.join(DATASET_FILES, dataset_name)
    for f in os.listdir(data_dir):
        file = os.path.join(data_dir, f)
        if os.path.isfile(file):
            if file.endswith('.pkl'):
                return _load_from_pkl(file)[0]
            elif file.endswith('.csv'):
                 return pd.read_csv(file, nrows=1)
    print("None found")


def _extract_num_from_string(string):
    try:
        return int(re.findall(r'\d+', string)[0])
    except:
        print("Age could not be extracted.")
        return None


def _extract_values_dict(d, cpgs):
    return list(itemgetter(*cpgs)(d))


def _extract_values_pandas(pd_df, cpgs):
    values = pd_df['VALUE'][pd_df['ID_REF'].isin(cpgs)]
    try:
        return [float(val) for val in values.tolist()]
    except:
        return [float(val[1]) for val in values.tolist()]


def _reduce_values_pandas(pd_df, cpgs):
    return pd_df[pd_df['ID_REF'].isin(cpgs)]


def _extract_age_pandas(pd_metadata):
    assert isinstance(pd_metadata, list)
    for entry in pd_metadata:
        entry = entry.lower()
        if "age" in entry:
            if "newborn" in entry:
                return 0
            elif "month" in entry:
                return (_extract_num_from_string(entry))/12.
            else:
                return _extract_num_from_string(entry)


def _extract_cpgs_pandas(pd_df):
    return set(pd_df['ID_REF'])


def _extract_values_pandas_manual(pd_df, cpgs):
    return pd_df[pd_df.columns.intersection(cpgs)].values.tolist()


def find_common_cpg(codes):
    def _get_keys(data):
        if isinstance(data, dict):
            return _extract_cpgs_pandas(data["data"])
        elif isinstance(data, pd.DataFrame):
            return list(data.columns.values)[3:]  # first three are age, is_male, is_case

    all_cpgs = [set(_get_keys(load_only_one(name_from_code(code)))) for code in codes]
    cpg = all_cpgs[0]
    for next_cpg, code in zip(all_cpgs[1:], codes[1:]):
        prev_l = len(cpg)
        cpg &= next_cpg
        print("Common CpG reduced from {} to {}".format(prev_l, len(cpg)))

    for original_cpg in all_cpgs:
        assert cpg.issubset(original_cpg)
    cpg = list(cpg)
    print("In total, {} cpg sites were found in common. Saving...".format(len(cpg)))
    with open(os.path.join(DATASET_FILES, 'common_cpg.pkl'), 'wb') as output:
        pickle.dump(cpg, output)


def check_relevant_cpgs(codes):
    def _get_keys(data):
        if isinstance(data, pd.DataFrame):
            return _extract_cpgs_pandas(data)
        elif isinstance(data, dict):
            print("Dataset {} is manual".format(codes))
            return data.keys()

    all_cpgs = [set(_get_keys(load_only_one(name_from_code(code))[0]["data"])) for code in codes]
    relevant_cpgs = set(_open_relevant_cpg())
    for cpg, code in zip(all_cpgs, codes):
        print("{} out of 1000 found in {}".format(len(cpg & relevant_cpgs), name_from_code(code)))


def _open_relevant_cpg():
    with open(os.path.join(DATASET_FILES, "relevant_cpg.json"), 'r') as f:
        list_of_cpg = f.read()
    return ast.literal_eval(list_of_cpg)


def _open_common_cpg():
    return _load_from_pkl(os.path.join(DATASET_FILES, "common_cpg.pkl"))


def _common_and_relevant():
    common = _open_common_cpg()
    relevant = _open_relevant_cpg()
    return list(set(common) & set(relevant))


def curate_dataset_internally(dataset_code):
    def _get_keys(data):
        if isinstance(data, pd.DataFrame):
            return _extract_cpgs_pandas(data)
        elif isinstance(data, dict):
            return data.keys()

    dataset_name = name_from_code(dataset_code)
    for case in (True, False):
        dataset = load_dataset(dataset_name, healthy=(not case), case=case)
        cpg = set(_get_keys(dataset[0]["data"]))
        init_num = len(cpg)
        for patient in dataset[1:]:
            cpg = cpg & set(_get_keys(patient["data"]))
        final_num = len(cpg)
        print("Dataset {} reduced from {} to {} CpGs".format(dataset_name, init_num, final_num))
        for patient in dataset:
            patient['data'] = _reduce_values_pandas(patient['data'], cpg)
        if case:
            name = "_case"
        else:
            name = "_control"
        with open(os.path.join(DATASET_FILES, 'curated_{}{}.pkl'.format(dataset_name, name)), 'wb') as output:
            pickle.dump(cpg, output)


def combine_healthy_datasets(dataset_names, paper_cpg=True):
    data_path = DATASET_FILES
    if paper_cpg:
        list_of_cpg = _common_and_relevant()
    else:
        list_of_cpg = _open_common_cpg()
    print("Reducing records to {} CpGs".format(len(list_of_cpg)))
    record_file = os.path.join(data_path, 'healthy1000.tfrecords')
    patients_counter = 0
    success_counter = 0
    with tf.io.TFRecordWriter(record_file) as writer:
        for name in dataset_names:
            dataset = load_dataset(name_from_code(name), healthy=True, case=False, specific_fields=list_of_cpg)
            if isinstance(dataset, List):
                for patient in dataset:
                    patients_counter += 1
                    data = _extract_values_pandas(patient['data'], list_of_cpg)
                    age = _extract_age_pandas(patient['metadata'])
                    if np.isnan(np.array(data)).any():
                        print("Data for patient {} in {} had NaN".format(patients_counter, name_from_code(name)))
                    elif age is None:
                        print("patient {} in {} had no age info".format(patients_counter, name_from_code(name)))
                    else:
                        success_counter += 1
                        writer.write(serialize_example(age=age,
                                                       data=data))
            elif isinstance(dataset, pd.DataFrame):
                for index, patient in dataset.iterrows():
                    age = patient['age']
                    is_case = bool(patient["is_case"])
                    if name.lower() == "hiv" and is_case:
                        pass
                    else:
                        data = patient.drop(["age", "is_male", "is_case"]).values
                        patients_counter += 1
                        if np.isnan(np.array(data)).any():
                            print("Data for patient {} in {} had NaN".format(patients_counter, name_from_code(name)))
                        elif age is None:
                            print("patient {} in {} had no age info".format(patients_counter, name_from_code(name)))
                        else:
                            success_counter += 1
                            writer.write(serialize_example(age=age,
                                                           data=data.tolist()))
    print("Data for {} patients out of {} was written.".format(success_counter, patients_counter))


def _example_proto(age, data):
    return tf.train.Example(features=tf.train.Features(feature={
        "age": tf.train.Feature(float_list=tf.train.FloatList(value=[tf.cast(age, tf.float32)])),
        "data": tf.train.Feature(float_list=tf.train.FloatList(value=tf.cast(data, tf.float32)))
    }
    ))


def serialize_example(age, data):
    return _example_proto(age, data).SerializeToString()


def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, features={
        "age": tf.io.FixedLenFeature([], tf.float32),
        "data": tf.io.FixedLenFeature(901, tf.float32)})


def tf_healthy_dataset():
    raw_ds_train = tf.data.TFRecordDataset([os.path.join(DATASET_FILES, 'healthy900_train.tfrecords')])
    raw_ds_test = tf.data.TFRecordDataset([os.path.join(DATASET_FILES, 'healthy900_test.tfrecords')])
    return raw_ds_train.map(_parse_function), raw_ds_test.map(_parse_function)


def tf_serialize_example(age, data):
    tf_string = tf.py_function(
        serialize_example,
        (age, data),
        tf.string)
    return tf.reshape(tf_string, ())


def deserialize_example(serialized_example):
    return tf.train.Example.FromString(serialized_example)


def create_dataset(data_code):
    dataset_name = name_from_code(data_code)
    destination_dir = os.path.join(DATASET_FILES, dataset_name)
    full_data = GEOparse.get_GEO(geo=data_code, destdir=DATASET_FILES)
    case_data = list()
    control_data = list()
    for patient in full_data.gsms.values():
        if any(s in info.lower() and "failed" not in info.lower() for info in patient.metadata['characteristics_ch1']
               for s in ["healthy", "control",
                         "ards: 0",
                         "non-gdm",
                         "baseline",
                         "normal",
                         "norm",
                         "current cscc: no",
                         'fetal intolerance: 1',
                         'case/control: 1',
                         "hc",
                         "s1",
                         "ctl"]):
            control_data.append({"metadata": patient.metadata['characteristics_ch1'],
                                 "data": patient.table})
        else:
            case_data.append({"metadata": patient.metadata['characteristics_ch1'],
                              "data": patient.table})

    print("{} case and {} control samples were collected. Pickling...".format(len(case_data), len(control_data)))
    os.makedirs(destination_dir, exist_ok=True)
    with open(os.path.join(destination_dir, dataset_name + '_case.pkl'), 'wb') as output:
        pickle.dump(case_data, output)
    with open(os.path.join(destination_dir, dataset_name + '_control.pkl'), 'wb') as output:
        pickle.dump(control_data, output)


def manual_create_dataset(data_code, cohorts=True):
    name_of_file = data_code + "_series_matrix.txt"
    name_normalized = data_code + "_matrix_normalized.txt"
    if name_of_file in os.listdir(DATASET_FILES):
        with open(os.path.join(DATASET_FILES, name_of_file), 'r') as f:
            init_data = [[j for j in i.split('\t')] for i in
                         f.read().splitlines()]
            print("Data collected from text file.")
    elif name_normalized in os.listdir(DATASET_FILES):
        with open(os.path.join(DATASET_FILES, name_normalized), 'r') as f:
            init_data = [[j for j in i.split('\t')] for i in
                         f.read().splitlines()]
            print("Data collected from text file.")
    else:
        name_of_file = name_of_file + '.gz'
        with open(os.path.join(DATASET_FILES, name_of_file), 'rb') as gz:
            init_data = [[j for j in i.split('\t')] for i in
                         (gz.read().decode('ISO-8859-1')).splitlines()]
            print("Data collected from .gz compression file.")
    dataset_name = name_from_code(data_code)
    destination_dir = os.path.join(DATASET_FILES, dataset_name)
    control_data = list()
    case_data = list()
    cohort_idx = 60
    ages_idx = 63
    genes_start_idx = 88
    data_counter = 0
    while len(init_data[ages_idx]) > 1:
        age = init_data[ages_idx].pop()
        if "newborn" in age.lower():
            age = '0'
        if cohort_idx != 0:
            cohort = init_data[cohort_idx].pop()
        else:
            cohort = None
        all_genes = dict()
        for gene in init_data[genes_start_idx:]:
            gene_name = re.search(r'cg[\d]{8}', gene[0])
            if gene_name:
                try:
                    all_genes[gene_name.group()] = float(gene.pop())
                except:
                    gene.pop()
                    all_genes[gene_name.group()] = None
        if cohorts:
            if "control" in cohort.lower() or "healthy" in cohort.lower():
                control_data.append({"age": _extract_num_from_string(age),
                                     "data": all_genes})
            elif "case" in cohort.lower() or "patient" in cohort.lower() or "syndrome" in cohort.lower():
                case_data.append({"age": _extract_num_from_string(age),
                                  "data": all_genes})
            else:
                print("Neither case or control found. Skipping entry")
        else:
            control_data.append({"age": _extract_num_from_string(age),
                                 "data": all_genes})
        os.makedirs(destination_dir, exist_ok=True)
        if len(case_data) + len(control_data) == 50:
            if cohorts:
                with open(os.path.join(destination_dir, dataset_name + "_case_" + str(data_counter) + '.pkl'),
                          'wb') as output:
                    pickle.dump(case_data, output)
            with open(os.path.join(destination_dir, dataset_name + "_control_" + str(data_counter) + '.pkl'),
                      'wb') as output:
                pickle.dump(control_data, output)
            data_counter += 1
            print(
                "{} data points for {} case and {} control patients was collected".format(len(control_data[0]["data"]),
                                                                                          len(case_data),
                                                                                          len(control_data)))
            case_data = list()
            control_data = list()

        with open(os.path.join(destination_dir, dataset_name + "_case_" + str(data_counter) + '.pkl'), 'wb') as output:
            pickle.dump(case_data, output)
        with open(os.path.join(destination_dir, dataset_name + "_control_" + str(data_counter) + '.pkl'),
                  'wb') as output:
            pickle.dump(control_data, output)


def manual_create_dataset_depression(data_code):
    name_of_file = data_code + "_series_matrix.txt"
    name_normalized = data_code + "_matrix_normalized.txt"
    if name_of_file in os.listdir(DATASET_FILES):
        with open(os.path.join(DATASET_FILES, name_of_file), 'r') as f:
            init_data_info = [[j for j in i.split('\t')] for i in
                              f.read().splitlines()]
            print("Data collected from text file.")
    if name_normalized in os.listdir(DATASET_FILES):
        with open(os.path.join(DATASET_FILES, name_normalized), 'r') as f:
            init_data_cpg = [[j for j in i.split('\t')] for i in
                             f.read().splitlines()]
            print("Data collected from text file.")

    init_data_cpg[0].pop(0)
    init_data_cpg.pop()  # as last cpg is shorter: len = 807, others are 1398
    for i, sample in enumerate(init_data_cpg[0]):
        init_data_cpg[0][i] = _extract_num_from_string(sample)
    cpg_names_list = ["age", "is_male", "is_case"]
    for entry in init_data_cpg[1:]:
        entry.pop(0)
        cpg_names_list.append(entry.pop(0))

    for i in range(len(init_data_info[25][1:])):
        init_data_info[25][i + 1] = _extract_num_from_string(init_data_info[25][i + 1])  # sample num
        init_data_info[35][i + 1] = _extract_num_from_string(init_data_info[35][i + 1])  # age

    is_male = ["M" in patient for patient in init_data_info[36][1:]]
    is_case = ["case" in patient.lower() for patient in init_data_info[34][1:]]

    init_data_info = np.squeeze(np.array([[init_data_info[25][1:],
                                init_data_info[35][1:],
                                is_male,
                                is_case]]))
    init_data_info = init_data_info[:, init_data_info[0].argsort()]
    init_data_cpg = np.squeeze(np.array(init_data_cpg))[:, ::2]
    init_data_cpg = init_data_cpg[:, init_data_cpg[0].astype(int).argsort()]
    init_data_cpg = np.concatenate((init_data_info[1:], init_data_cpg[1:]))
    del init_data_info
    init_data_cpg = pd.DataFrame(data=init_data_cpg.T, columns=cpg_names_list)
    init_data_cpg = init_data_cpg.apply(pd.to_numeric, errors='coerce')

    dataset_name = name_from_code(data_code)
    destination_dir = os.path.join(DATASET_FILES, dataset_name)

    os.makedirs(destination_dir, exist_ok=True)
    init_data_cpg.to_csv(os.path.join(destination_dir, dataset_name + ".csv"))


if __name__ == '__main__':
    datasets = name_from_code('all')
    # find_common_cpg(datasets)
    combine_healthy_datasets(datasets)
    # for code in datasets:
    #     create_dataset(data_code=code)
    #     curate_dataset_internally(code)
    # manual_create_dataset_depression(data_code=name_from_code("depression"))