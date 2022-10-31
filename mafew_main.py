import csv
from enum import Enum

import numpy as np
import pandas as pd
import os

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from xgboost import XGBClassifier

DATA_PATH = r"F:\data\epigenetics v2\ewasdatahub"


class Diseases(Enum):
    Alzheimers = "Alzheimer's disease"
    Asthma = "asthma"
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


def save_disease(disease):
    names = pd.read_csv(os.path.join(DATA_PATH, "disease_methylation_v1.txt"), sep='\t', nrows=2)
    names = names.to_numpy().squeeze()
    columns = np.concatenate((np.array([0]),
                              np.argwhere((names[0] == disease.value) & (names[1] == "whole blood")).squeeze()))
    with open(os.path.join(DATA_PATH, "disease_methylation_v1.txt"), 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        list_of_cgs = []
        for row in tqdm(reader):
            list_of_cgs.append(np.array(row)[columns].squeeze())
    to_df = pd.DataFrame(list_of_cgs)
    path = os.path.join(DATA_PATH, "diseases")
    os.makedirs(path, exist_ok=True)
    to_df.to_csv(os.path.join(path, disease.value + ".csv"), index=False)


def clean_disease(disease):
    path = os.path.join(DATA_PATH, "diseases", disease.value + ".csv")
    df = pd.read_csv(path, low_memory=False)
    nan_thresh = np.ceil(0.95*df.shape[1]).astype(int)
    df = df.dropna(axis=0, thresh=nan_thresh)
    df = df.dropna(axis=1, thresh=nan_thresh)
    df = df[~df["0"].astype("str").str.startswith("ch")]
    df = df.fillna(0)
    df.to_csv(os.path.join(DATA_PATH, "diseases", disease.value + "_clean.csv"), index=False)


def load_disease(disease):
    mom4i_cpg = pd.read_csv(os.path.join(DATA_PATH, "diseases", "diabetes_cpg_list.csv"))
    df = pd.read_csv(os.path.join(DATA_PATH, "diseases", disease.value + "_clean.csv"))
    common = np.hstack((np.array(["sample_type"]), np.array(np.intersect1d(mom4i_cpg["Name"], df["0"])).astype(str)))
    return df[df["0"].isin(common)]


def train_xgb(df):
    # X = df.drop("0", axis=1)
    X = df.drop(3, axis=0).transpose()
    y = df.drop("0", axis=1).loc[3]
    to_binary = {"control": 0,
                 "case": 1}
    y = y.map(to_binary).astype(int)
    X.columns = X.iloc[0]
    X = X[1:].astype(float)
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # fit model no training data
    model = XGBClassifier()
    model.fit(X_train, y_train)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy of diabetes model: %.2f%%" % (accuracy * 100.0))


if __name__ == '__main__':
    train_xgb(load_disease(Diseases.T2D))
