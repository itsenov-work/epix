import csv
from enum import Enum

import numpy as np
import pandas as pd
import os

from tqdm import tqdm

DATA_PATH = r"C:\projects\epigenetics\data\raw"


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

    # data = pd.read_csv(os.path.join(DATA_PATH, "disease_methylation_v1.txt"), sep='\t', usecols=columns[:2])
    # if save:
    #     data.to_csv()
    # else:
    #     path = os.path.join(DATA_PATH, "diseases")
    #     os.makedirs(path, exist_ok=True)
    #     return data.to_csv(os.path.join(path, disease.value + ".csv"))


if __name__ == '__main__':
    save_disease(Diseases.T2D)
