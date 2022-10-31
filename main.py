import csv
import json
import shutil
import time

import pandas as pd
import os
import os.path as osp
import pickle
import fastparquet
import tqdm
import xgboost

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

epix_data_path = r'C:\Users\itsen\workspaces\epix\epigenetic data\data\epigenetics'


def get_dataset_path(dataset_name):
    return osp.join(epix_data_path, dataset_name)


def get_csv_path(dataset_name):
    return osp.join(get_dataset_path(dataset_name), f'{dataset_name}.csv')


def get_sample_folder(dataset_name):
    dataset_path = get_dataset_path(dataset_name)
    return osp.join(dataset_path, 'samples')


def get_samples(dataset_name):
    sample_folder = get_sample_folder(dataset_name)
    return os.listdir(sample_folder)


def get_column_names_path(dataset_name):
    sample_folder = get_sample_folder(dataset_name)
    return osp.join(sample_folder, f'column_names.csv')


def get_column_names(dataset_name):
    with open(get_column_names_path(dataset_name), 'r') as f:
        return f.read()


def save_sample(dataset_name, line, sample_num, column_names):
    try:
        sample_folder = get_sample_folder(dataset_name)
        path = osp.join(sample_folder, f'sample_{sample_num}.csv')
        if osp.exists(path):
            os.remove(path)
        with open(path, 'w+') as f:
            f.writelines([column_names, line])
        print(f"Saved sample {sample_num}")
    except Exception as e:
        print(e)
        print(f"Failed to save sample {sample_num}")


def save_column_names(dataset_name, column_names):
    try:
        path = get_column_names_path(dataset_name)
        if osp.exists(path):
            os.remove(path)
        with open(path, 'w+') as f:
            f.write(column_names)
        print(f"Saved column names")
    except Exception as e:
        print(e)
        print(f"Failed to save column names.")



def samplify(dataset_name):
    sample_folder = get_sample_folder(dataset_name)
    os.makedirs(sample_folder, exist_ok=True)

    csv_file = get_csv_path(dataset_name)
    column_names = None
    with open(csv_file) as fp:
        for i, line in enumerate(fp):
            if i == 0:
                column_names = line
                save_column_names(dataset_name, column_names)
            else:
                save_sample(dataset_name, line, i - 1, column_names=column_names)


def get_data(dataset_name):
    is_case = list()
    is_male = list()
    cpg_data = list()
    csv_path = get_csv_path(dataset_name)
    missing_cgs = list()
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        cpg_headers = [h for h in headers if h.startswith('cg')]
        for i, row in tqdm.tqdm(enumerate(reader)):
            is_case.append(row["is_case"])
            is_male.append(row["is_male"])

            l = list()
            for cpg in cpg_headers:
                try:
                    l.append(float(row[cpg]))
                except Exception as e:
                    missing_cgs.append(cpg)
                    l.append(0)

            cpg_data.append(l)
    print("Missing CpGs: ", len(missing_cgs))
    return cpg_headers, cpg_data, is_case, is_male


def pickle_path(dataset_name):
    return osp.join(get_dataset_path(dataset_name), "cpgs_and_case.pickle")


def pickle_dataset(dataset_name):
    import time
    t0 = time.time()
    data = get_data(dataset_name)
    t1 = time.time()
    print(f"Time elapsed: {t1 - t0}")

    import pickle
    with open(pickle_path(dataset_name), 'wb+') as f:
        pickle.dump(data, f)
    t2 = time.time()
    print(f"Pickle time: {t2 - t1}")


def load_pickle(dataset_name):
    t0 = time.time()
    with open(pickle_path(dataset_name), 'rb') as f:
        p = pickle.load(f)
    t1 = time.time()
    print(f"Pickle load time: {t1 - t0}")

    return p


def load_controls(dataset_name):
    t0 = time.time()
    with open(osp.join(get_dataset_path(dataset_name), f'{dataset_name}_control.pkl'), 'rb') as f:
        p = pickle.load(f)
    t1 = time.time()
    print(f"Pickle load time: {t1 - t0}")
    return p


def save_json(folder, filename, obj):
    with open(osp.join(folder, f"{filename}.json"), 'w+') as f:
        json.dump(obj, f)


def get_top_n_features(dataset_name, X, y, cpg_headers, num_features, prefix=""):
    select_k_best = SelectKBest(chi2, k=num_features)
    X1 = select_k_best.fit_transform(X, y)
    top_n_cpg = select_k_best.get_feature_names_out(input_features=cpg_headers)

    clf = ExtraTreesClassifier(n_estimators=100)
    clf = clf.fit(X1, y)

    model = SelectFromModel(clf, prefit=True)
    X1 = model.transform(X1)
    relevant_cpg = model.get_feature_names_out(input_features=top_n_cpg)

    folder = osp.join(get_dataset_path(dataset_name), f'feature_selection_{prefix}_{num_features}')
    os.makedirs(folder, exist_ok=True)

    save_json(folder, f'top{num_features}_cpg', top_n_cpg.tolist())
    save_json(folder, f'top{num_features}_cpg_scores',
              sorted(
                [[k, select_k_best.scores_[i]] for i, k in enumerate(cpg_headers) if k in top_n_cpg],
                key=lambda x: x[1], reverse=True))
    save_json(folder, f'relevant_cpgs_{num_features}', relevant_cpg.tolist())
    return X1, relevant_cpg


if __name__ == '__main__':
    dataset_name = "food_allergy"

    pickle_dataset(dataset_name)

    cpg_headers, cpg_data, is_case, is_male = load_pickle(dataset_name)

    X = cpg_data
    y = [int(i == 'True') for i in is_case]
    if all([k == 1 for k in y]):
        print("ALL y are cases. Aborting!")
        exit()
    if all([k == 0 for k in y]):
        print("NO y are cases. Aborting!")
        exit()

    t0 = time.time()
    X = pd.DataFrame(cpg_data, columns=[cpg_headers])
    t1 = time.time()
    print(f"Dataframe ready! {t1 - t0}")

    X_new, relevant_cpg = get_top_n_features(dataset_name, X, y, cpg_headers, num_features=10000)
    X_new, relevant_cpg = get_top_n_features(dataset_name, X_new, y, relevant_cpg, num_features=1000, prefix="after10000")

    from numpy import loadtxt
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # split data into train and test sets
    seed = 9
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=test_size, random_state=seed)
    # fit model no training data
    model = XGBClassifier()
    model.fit(X_train, y_train)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy of first model: %.2f%%" % (accuracy * 100.0))

    model = SelectFromModel(model, prefit=True)
    X_new = model.transform(X_new)
    relevant_cpg = model.get_feature_names_out(input_features=relevant_cpg)
    folder = osp.join(get_dataset_path(dataset_name), f'feature_selection_best')
    os.makedirs(folder, exist_ok=True)
    save_json(folder, f'{dataset_name}_cpg', relevant_cpg.tolist())
    # import numpy as np
    # from sklearn.linear_model import RidgeCV
    #
    # ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X_new, y)
    # importance = np.abs(ridge.coef_)
    # feature_names = np.array(cpg_headers)

    from sklearn.feature_selection import SequentialFeatureSelector


    # tic_fwd = time()
    # sfs_forward = SequentialFeatureSelector(
    #     ridge, n_features_to_select=2, direction="forward"
    # ).fit(X_new, y)
    # toc_fwd = time()

    # tic_bwd = time.time()
    # sfs_backward = SequentialFeatureSelector(
    #     ridge, n_features_to_select=100, direction="backward"
    # ).fit(X_new, y)
    # toc_bwd = time.time()
    # #
    # # print(
    # #     "Features selected by forward sequential selection: "
    # #     f"{feature_names[sfs_forward.get_support()]}"
    # # )
    # # print(f"Done in {toc_fwd - tic_fwd:.3f}s")
    # print(
    #     "Features selected by backward sequential selection: "
    #     f"{feature_names[sfs_backward.get_support()]}"
    # )
    # print(f"Done in {toc_bwd - tic_bwd:.3f}s")