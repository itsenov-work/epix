import csv
import json
import time
from datetime import timedelta
from enum import Enum

import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt
from sklearn.feature_selection import RFECV, SelectKBest, chi2
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm
import xgboost as xgb
from boruta import BorutaPy

DATA_PATH = r"F:\data\epigenetics v2\ewasdatahub"


class Diseases(Enum):
    # Alzheimers = "Alzheimer's disease"
    # Asthma = "asthma"
    AutismSpectrum = 'autism spectrum disorder'
    ChildhoodAsthma = 'childhood asthma'
    Crohns = "Crohn's disease"
    Down = 'Down syndrome'
    Graves = "Graves' disease"
    Huntingtons = "Huntington's disease"
    CognitalAnomalies = 'intellectual disability and congenital anomalies'
    Kabuki = 'Kabuki syndrome'
    MS = 'multiple sclerosis'
    NephrogenicRest = 'nephrogenic rest'
    Panic = 'panic disorder'
    Parkinsons = "Parkinson's disease"
    Preeclampsia = 'preeclampsia'
    Psoriasis = 'psoriasis'
    RaspiratoryAllergy = 'respiratory allergy'
    RheumatoidArthritis = 'rheumatoid arthritis'
    Schizophrenia = 'schizophrenia'
    SilverRussel = 'Silver Russell syndrome'
    Sjorgens = "Sjogren's syndrome"
    Spina = 'spina bifida'
    Stroke = 'stroke'
    InsulinResist = 'systemic insulin resistance'
    Lupus = 'systemic lupus erythematosus'
    Sclerosis = 'systemic sclerosis'
    T2D = 'type 2 diabetes'
    UlcerativeColitis = 'Ulcerative colitis'


class Conditions(Enum):
    Age = "age"
    Sex = "sex"
    BMI = "bmi"


def save_factor(factor):
    if isinstance(factor, Diseases):
        data_path = os.path.join(DATA_PATH, "disease_methylation_v1.txt")
        names = pd.read_csv(data_path, sep='\t', nrows=2)
        names = names.to_numpy().squeeze()
        columns = np.concatenate((np.array([0]),
                                  np.argwhere((names[0] == factor.value) & (names[1] == "whole blood")).squeeze()))
    elif isinstance(factor, Conditions):
        data_path = os.path.join(DATA_PATH, factor.value + "_methylation.txt")
        names = pd.read_csv(data_path, sep='\t', nrows=2)
        names = names.to_numpy().squeeze()
        columns = np.concatenate((np.array([0]),
                                  np.argwhere(names[1] == "whole blood").squeeze()))
    else:
        return
    if columns.size < 10:
        return
    if factor.value == "age":
        column_chunks = np.array_split(columns, 5)
    else:
        column_chunks = np.array_split(columns, 3)
    for i, col in enumerate(column_chunks):
        print(f"Section {i}...")
        with open(data_path, 'r') as f:
            reader = csv.reader(f, delimiter="\t")
            list_of_cgs = []
            for row in tqdm(reader, total=485000):
                list_of_cgs.append(np.array(row)[columns].squeeze())
        to_df = pd.DataFrame(list_of_cgs)
        del list_of_cgs
        if isinstance(factor, Diseases):
            save_path = os.path.join(DATA_PATH, "diseases")
        elif isinstance(factor, Conditions):
            save_path = os.path.join(DATA_PATH, "conditions")
        else:
            return
        # os.makedirs(save_path, exist_ok=True)
        to_df.to_csv(os.path.join(save_path, factor.value + "_{}.csv".format(i)), index=False)
        del to_df



def save_condition(condition):
    tissues = pd.read_csv(os.path.join(DATA_PATH, condition.value + "_methylation.txt"), sep='\t',
                          nrows=2, low_memory=False)
    selection = (tissues.loc[1] == "whole blood").to_numpy()
    del tissues
    df = pd.read_csv(os.path.join(DATA_PATH, condition.value + "_methylation.txt"), sep='\t', low_memory=False)
    path = os.path.join(DATA_PATH, "conditions")
    df = df.loc[:, selection]
    df.to_csv(os.path.join(path, condition.value + ".csv"), index=False)


def save_all_diseases():
    names = pd.read_csv(os.path.join(DATA_PATH, "disease_methylation_v1.txt"), sep='\t', nrows=2)
    names = names.to_numpy().squeeze()
    for disease in Diseases:
        columns = np.concatenate((np.array([0]),
                                  np.argwhere((names[0] == disease.value) & (names[1] == "whole blood")).squeeze()))
    data = pd.read_csv(os.path.join(DATA_PATH, "disease_methylation_v1.txt"), sep='\t', )
    tgt_path = os.path.join(DATA_PATH, "auto_generated")
    os.makedirs(tgt_path, exist_ok=True)
    for disease in tqdm(Diseases):
        criteria = (data.loc[0] == disease.value) & (data.loc[1] == "whole blood")
        # criteria[0] = 1
        data.loc[:, criteria].to_csv(
            os.path.join(tgt_path, disease.value + ".csv"), index=False)


def clean_factor(factor):
    if isinstance(factor, Diseases):
        path = os.path.join(DATA_PATH, "diseases", factor.value + ".csv")
    elif isinstance(factor, Conditions):
        path = os.path.join(DATA_PATH, "conditions", factor.value + ".txt")
    else:
        print("Warning: passed criteria is not recognised.")
        path = os.path.join(DATA_PATH, factor.value + ".csv")
    try:
        if isinstance(factor, Diseases):
            df = pd.read_csv(path, low_memory=False, nrows=10)
            saving_path = get_disease_path(factor)
        elif isinstance(factor, Conditions):
            df = pd.read_csv(path, sep='\t', nrows=10)
            saving_path = get_condition_path(factor)
        else:
            saving_path = os.path.join(DATA_PATH, str(factor.value), factor.value + "_clean.csv")
    except FileNotFoundError as FNF:
        print(FNF)
        return
    with open(os.path.join(saving_path, 'cleaning_info.txt'), 'w+') as f:
        nan_cnt = df.isna().sum().sum()
        f.write(f"Initial data: Total patients were: {df.shape[1] - 1}, \n")
        if isinstance(factor, Diseases):
            cases = (df.loc[3] == 'case').T.sum()
            control = (df.loc[3] == 'control').T.sum()
            f.write(f" with {cases} cases and {control} controls. \n")
        f.write(f"Initial data: Total NaNs were: {nan_cnt}, or {100 * nan_cnt / df.size} % of data. \n")
        f.write("Initial data: Total CpG number was: {}. \n".format(df["0"].astype("str").str.startswith("ch").size))
        cpg_nan_thresh = 0.8
        patient_nan_thresh = 0.9
        df = df.dropna(axis=0, thresh=np.ceil(cpg_nan_thresh * (df.shape[1] + 1)).astype(int))  # cpgs
        df = df.dropna(axis=1, thresh=np.ceil(patient_nan_thresh * (df.shape[0] + 3)).astype(int))  # patients
        if df.empty:
            print("Exhausted data.")
            return
        df = df[~df["0"].astype("str").str.startswith("ch")]
        nan_cnt = df.isna().sum().sum()
        df = df.fillna(0)
        df.to_csv(os.path.join(saving_path, factor.value + "_clean.csv"), index=False)
        f.write(f"Clean data: Total patients are: {df.shape[1] - 1}, \n")
        if isinstance(factor, Diseases):
            cases = (df.loc[3] == 'case').T.sum()
            control = (df.loc[3] == 'control').T.sum()
            f.write(f" with {cases} cases and {control} controls. \n")
        f.write(f"Clean data: Total NaNs are: {nan_cnt}, or {100 * nan_cnt / df.size} % of data. \n")
        f.write("Clean data: Total CpG number is: {}. \n".format(df["0"].astype("str").str.startswith("ch").size))
    with open(os.path.join(saving_path, 'dataset_info.json'), "w+") as d:
        info_as_dict = {"cases": int(cases),
                        "controls": int(control)}
        json.dump(info_as_dict, d, indent=4)


def load_raw(disease):
    mom4i_cpg = pd.read_csv(os.path.join(DATA_PATH, "diseases", "diabetes_cpg_list.csv"))
    df = pd.read_csv(os.path.join(DATA_PATH, "diseases", disease.value + "_clean.csv"))
    common = np.hstack((np.array(["sample_type"]), np.array(np.intersect1d(mom4i_cpg["Name"], df["0"])).astype(str)))
    return df[df["0"].isin(common)]


def load_clean(disease):
    path = get_disease_path(disease)
    if path is None:
        return
    data_loc = os.path.join(path, disease.value + "_clean.csv")
    return pd.read_csv(data_loc, low_memory=False, index_col=0)


def get_disease_path(disease):
    folder = os.path.join(DATA_PATH, "diseases", disease.value)
    if os.path.isdir(folder):
        return folder
    else:
        return None


def get_condition_path(condition):
    folder = os.path.join(DATA_PATH, "conditions", condition.value)
    if os.path.isdir(folder):
        return folder
    else:
        return None


def create_disease_path(disease):
    folder = os.path.join(DATA_PATH, "diseases", disease.value)
    os.makedirs(folder, exist_ok=True)


def prepare_dataset(df, test_split=0.2):
    df.columns = df.loc["sample_id"]  # should have saved with index=False when cleaning data, I think
    df = df[1:]
    X = df.drop(["disease", "tissue", "sample_type"], axis=0).transpose()
    X = X.astype(float)

    y = df.loc["sample_type"]
    to_binary = {"control": 0,
                 "case": 1}
    y = y.map(to_binary).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
    return X_train, X_test, y_train, y_test

    # fit model no training data


def find_n_features(begin, minimum, step):
    assert 0 < step < 1
    n_features = [begin]
    while n_features[-1] >= minimum:
        n_features.append(n_features[-1] * (1 - step))
    n_features[-1] = minimum
    return n_features


def create_xgb_model(disease, save=True):
    """ Prepare dataset: """
    folder = None
    model_path = None
    if save:
        folder = get_disease_path(disease)
        if folder is None:
            return
        model_path = os.path.join(folder, disease.value + "_xgb.json")
    dataset = load_clean(disease)
    if dataset is None:
        return
    print("Loaded Dataset...")
    X_train, X_test, y_train, y_test = prepare_dataset(dataset, 0.25)
    del dataset
    """ Select model: """
    model = xgb.XGBClassifier()
    """Perform feature selection: """
    minimum_features = 25
    # init_features = X_train.shape[0]
    # step = 0.1
    # feature_selector = RFECV(
    #     estimator=model,
    #     step=step,
    #     cv=StratifiedKFold(4),
    #     scoring="accuracy",
    #     min_features_to_select=minimum_features,
    # )
    initial_feature_selector = SelectKBest(chi2, k=8000)
    initial_feature_selector.fit(X_train, y_train)
    feature_idx = initial_feature_selector.get_support(indices=True)
    feature_names = X_train.columns[feature_idx]
    X_train = pd.DataFrame(initial_feature_selector.transform(X_train),
                           index=X_train.index, columns=feature_names)
    X_test = pd.DataFrame(initial_feature_selector.transform(X_test),
                          index=X_test.index, columns=feature_names)

    feature_selector = BorutaPy(
        estimator=model,
        n_estimators='auto',
        max_iter=20  # number of trials to perform
    )
    feature_selector.fit(np.array(X_train), np.array(y_train))
    #  Keep CpG names:
    # feature_idx = feature_selector.get_support(indices=True)
    feature_names = X_train.columns[feature_selector.support_]
    X_train = pd.DataFrame(feature_selector.transform(np.array(X_train)), index=X_train.index, columns=feature_names)
    X_test = pd.DataFrame(feature_selector.transform(np.array(X_test)), index=X_test.index, columns=feature_names)
    print("Optimal number of features : %d" % feature_selector.n_features_)
    if save:
        # Plot number of features VS. cross-validation scores
        # featuring_fig, featuring_ax = plt.subplots()
        # featuring_ax.set_xlabel("Number of features selected")
        # featuring_ax.set_ylabel("Cross validation score (accuracy)")
        # featuring_ax.plot(
        #     range(len(rfecv.grid_scores_)),
        #     rfecv.grid_scores_,
        # )
        # featuring_fig.savefig(os.path.join(folder, disease.value + "_feature_selection.png"))
        pd.Series(feature_names, name="CpG_sites").to_csv(
            os.path.join(folder, disease.value + "_selected_features.csv"), index=False)
    """Fit and save final model with stats: """

    # make predictions for test data
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy of {} model: {}".format(disease.value, np.round(accuracy * 100.0, 2)))
    classification_metrics = classification_report(y_test, predictions, output_dict=True)
    confusion_fig, confusion_ax = plt.subplots()
    features_fig, features_ax = plt.subplots()
    d = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
    d.plot(ax=confusion_ax)
    xgb.plot_importance(model, ax=features_ax, max_num_features=20)
    if save:
        model.save_model(model_path)
        confusion_fig.savefig(os.path.join(folder, disease.value + "_confusion_matrix_xgb.png"))
        features_fig.savefig(os.path.join(folder, disease.value + "_feature_importance_xgb.png"))
        with open(os.path.join(folder, disease.value + "_xgb_report.json"), "w") as report:
            json.dump(classification_metrics, report, indent=4)
    else:
        print("Full report: ", classification_metrics)
        plt.show()
    plt.close()


def main_training_loop():
    for disease in Diseases:
        print(f"Training model for {disease.name}...")
        t1 = time.time()
        create_xgb_model(disease)
        print(f"Finished in {timedelta(seconds=time.time() - t1)}. ")


def main_cleaning_loop():
    for factor in Conditions:
        print(f"Cleaning up {factor.name}...")
        t1 = time.time()
        clean_factor(factor)
        print(f"Finished in {timedelta(seconds=time.time() - t1)}. ")


def collect_for_plot():
    collected_info = {}
    for disease in tqdm(Diseases):
        folder = get_disease_path(disease)
        df = load_clean(disease)
        if df is None:
            continue
        cases = (df.loc["sample_type"] == 'case').T.sum()
        controls = (df.loc["sample_type"] == 'control').T.sum()
        with open(os.path.join(folder, disease.value + "_xgb_report.json"), "r") as report:
            training_info = json.load(report)
        isolated_cpgs_n = pd.read_csv(os.path.join(folder, disease.value + "_selected_features.csv")).size
        collected_info[disease.value] = {"Total_patients": int(cases + controls),
                                         "Cases": int(cases),
                                         "Controls": int(controls),
                                         "Num selected CpGs": int(isolated_cpgs_n),
                                         "Training accuracy": float(training_info["accuracy"]),
                                         "Training F1 score": float(training_info["weighted avg"]["f1-score"])}
    with open(os.path.join(DATA_PATH, "diseases", "all_info.json"), 'w') as f:
        json.dump(collected_info, f, indent=4)


def plots_from_info():
    import seaborn as sns
    sns.set_theme("talk")
    
    sns.set(style="darkgrid", font_scale=2)
    with open(os.path.join(DATA_PATH, "all_info.json"), "r") as f:
        info = json.load(f)
    acc_data = pd.DataFrame([[name, 100*acc["Training accuracy"]] for name, acc in zip(info.keys(), info.values())],
                            columns=["Disease", "Accuracy"])
    sns.barplot(x="Accuracy", y="Disease", data=acc_data,
                label="Classification accuracy", color="g", alpha=0.5)
    plt.xlim(50, 100)
    plt.show()


if __name__ == '__main__':
    for condition in Conditions:
        save_factor(condition)
