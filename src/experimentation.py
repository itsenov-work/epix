import json
import os
from datetime import datetime

import setuptools.glob
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from strenum import StrEnum

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from data.epix_methylation_db import EPIXMethylationDatabase
from mafew_main import DATA_PATH
from model_training import AvailableClassifiers, StatisticalFeatureSelect, ModelFeatureSelector, ModelTrainer, \
    XGBTrainer, CatTrainer
from utils.logging import LoggerMixin
import xgboost as xgb

from utils.logging import Logger
import seaborn as sns

sns.set_theme()
logger = Logger('Experimentalist')


class PathControl:
    def __init__(self, root):
        self.root = root
        self.experiments_path = os.path.join(self.root, "experiments")
        self.data_path = os.path.join(self.root, "samplified_diseases")
        self.path = None

    def get_path(self):
        return self.experiments_path

    def create_path(self, interpretation_config, model_config):
        diseases = [trait["disease"] for trait in interpretation_config if
                    trait["disease"] is not None]
        if len(diseases) > 2:
            name = "{}_traits_".format(len(diseases))
        elif len(diseases) == 0:
            name = "experiment_"
        else:
            name = ""
            for disease in diseases:
                name = name + "{}_".format(disease)
        name = name + model_config["model"]
        name = name + "_" + datetime.now().strftime('%Y%m%d')
        all_experiments = os.listdir(self.experiments_path)
        if any(name in other_experiment for other_experiment in all_experiments):
            name = name + "_" + str(sum(name in other_experiment for other_experiment in all_experiments) + 1)
        self.path = os.path.join(self.experiments_path, name)
        os.makedirs(self.path, exist_ok=False)

    def get_model_path(self):
        mpath = os.path.join(self.path, "model")
        os.makedirs(mpath, exist_ok=True)
        return mpath

    def get_data_path(self):
        return self.data_path

    def get_config_path(self):
        cpath = os.path.join(self.path, "configs")
        os.makedirs(cpath, exist_ok=True)
        return cpath

    def get_bio_path(self):
        path = os.path.join(self.root, "bio reports")
        os.makedirs(path, exist_ok=True)
        return path

    def get_data_stats_path(self):
        dpath = os.path.join(self.path, "data")
        os.makedirs(dpath, exist_ok=True)
        return dpath


class Experimentalist(PathControl):
    def __init__(self, path):
        super(Experimentalist, self).__init__(root=path)
        # Set some defaults:
        self.experiment_config = {"timestamp": self._get_timestamp(),
                                  "model": AvailableClassifiers.XGBoost,
                                  "test_split": 0.25,
                                  "feature_selection": [{"method": StatisticalFeatureSelect.anova_f,
                                                         "reduce_to": 8000},
                                                        {"method": ModelFeatureSelector.boruta,
                                                         "reduce_to": None}
                                                        ],
                                  }
        self.data_config = {}
        self.interpretation_config = {}
        self.interpretation_config = {}
        self.data_pipeline = None
        self.model_trainer = None
        self._connect_pipelines()

    def _connect_pipelines(self):
        self.data_pipeline = EPIXMethylationDatabase(self.get_data_path())
        self.model_trainer = ModelTrainer()

    @staticmethod
    def _get_timestamp():
        format_str = "%A, %d %b %Y %H:%M:%S %p"
        return datetime.now().strftime(format_str)

    def add_target(self, target: StrEnum, n: int = None):
        pass

    def add_non_target(self, non_target: StrEnum, n: int = None):
        pass

    def controls(self):
        pass

    def add_data_config(self, data_config, interpretation_config):
        self.data_config = data_config
        self.interpretation_config = interpretation_config

    def add_model_config(self, experiment_config):
        self.experiment_config = {**self.experiment_config, **experiment_config}

    def save_configs(self):
        for name, conf in zip(["data_config", "experiment_config", "interpretation_config"],
                              [self.data_config, self.experiment_config, self.interpretation_config]):
            with open(os.path.join(self.get_config_path(), name + ".json"), "w") as outfile:
                json.dump(conf, outfile, indent=4)
        logger.s("Configs recorded.")

    def configure_trainer(self):
        if self.experiment_config["model"] == AvailableClassifiers.XGBoost:
            self.model_trainer = XGBTrainer()
        elif self.experiment_config["model"] == AvailableClassifiers.CatBoost:
            self.model_trainer = CatTrainer()
        else:
            raise NotImplementedError("The selected model does not exist.")

        for selectors in self.experiment_config["feature_selection"]:
            self.model_trainer.add_feature_selection(selectors["method"], reduce_to_n=selectors["reduce_to"])

    def create_interpretation(self, meta):
        for patient, metadata in meta.items():
            meta[patient]["label"] = all(
                (metadata[criteria] == value for config in self.interpretation_config for criteria,
                                                                                          value in config.items()))
        return meta

    def get_mom4i_cpgs(self):
        with open(os.path.join(self.get_bio_path(),
                               "stats_COMMON_CPG_PROMOTER_RANGE_DIABETES_diabetes_fclassif.json")) as f:
            cpg_dict = json.load(f)
        self.data_config["cpgs"] = [each_cpg["cpg"] for each_cpg in cpg_dict]

    def get_dataset(self):
        df, meta = self.data_pipeline.get_data(self.data_config)
        meta = self.create_interpretation(meta)
        y = np.array([[gsm, meta[gsm]["label"]] for gsm in df["sample_id"]])
        df.index = df["sample_id"]
        df.drop("sample_id", axis=1, inplace=True)
        df = df.astype(float)

        to_binary = {"True": 1,
                     "False": 0}
        y = pd.Series(y[:, 1], index=y[:, 0])
        y = y.map(to_binary).astype(int)

        data = {"training_data": {},
                "testing_data": {}}

        (data["training_data"]["x"],
         data["testing_data"]["x"],
         data["training_data"]["y"],
         data["testing_data"]["y"]) = train_test_split(df, y, test_size=self.experiment_config["test_split"])
        for gsm in data["training_data"]["y"].index:
            meta[gsm]["training"] = 1
        for gsm in data["testing_data"]["y"].index:
            meta[gsm]["training"] = 0
        return data, meta

    def run(self):
        logger.start("Experiment commencing...")
        self.create_path(self.interpretation_config, self.experiment_config)
        self.get_mom4i_cpgs()
        self.configure_trainer()
        self.save_configs()
        data, meta = self.get_dataset()
        self.model_trainer.add_dataset(data)
        del data
        logger.s("Model and Data Pipeline configured successfully!")
        self.produce_data_stats(meta)
        logger.start("Starting Feature Selection process")

        self.model_trainer.feature_select()
        logger.end("Feature selection finished!")
        logger.start("Model training beginning...")
        self.model_trainer.train()
        self.model_trainer.save_model(os.path.join(self.get_model_path(), "saved_model.json"))
        self.produce_classification_stats()
        logger.s("Model saved!")
        logger.end()

    def produce_data_stats(self, meta):
        matching_keys = set.intersection(*[set(single_meta.keys()) for single_meta in meta.values()])
        matching_keys.remove("folder")
        matching_keys.remove("tissue")
        for key in matching_keys:
            plotted_stats = []
            plotted_colors = []
            # train_or_test = []
            for patient in meta.values():
                plotted_stats.append(patient[key])
                plotted_colors.append("target" if patient["label"] == 1 else "non-target")
                # train_or_test.append(int(patient["training"]))
            data_to_plot = pd.DataFrame({key: plotted_stats, "label": plotted_colors, })  # "training": train_or_test})
            data_plot = sns.countplot(data=data_to_plot, x=key, hue="label")
            plt.title("Patient Count for: " + key)
            handles, labels = data_plot.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            data_plot.legend(by_label.values(), by_label.keys())
            data_plot.get_figure().savefig(os.path.join(self.get_data_stats_path(), key + ".png"))
            plt.close()
            logger.s("Produced chart for {}".format(key))

    def produce_classification_stats(self):
        confusion_fig, confusion_ax = plt.subplots()
        features_fig, features_ax = plt.subplots()
        xgb.plot_importance(self.model_trainer.get_model(), ax=features_ax, max_num_features=5)
        features_fig.savefig(os.path.join(self.get_model_path(), "feature_importance.png"))
        d = ConfusionMatrixDisplay(self.model_trainer.get_confusion_matrix())
        d.plot(ax=confusion_ax)
        confusion_fig.savefig(os.path.join(self.get_model_path(), "confusion_matrix.png"))
        with open(os.path.join(self.get_model_path(), "performance_report.json"), "w") as report:
            json.dump(self.model_trainer.get_metrics_report(), report, indent=4)


if __name__ == '__main__':
    exp = Experimentalist(DATA_PATH)

    data_config = {
        "requests": [
            {
                "filters": {
                    "disease": "type 2 diabetes",
                    "tissue": "whole blood",
                },
            },
        ],
    }
    #
    # mom4i_cpg_config = [{"disease": "diabetes",
    #                     "location": "promoter"},
    # ]
    interpreter_config = [{
        'disease': "type 2 diabetes",
        'sample_type': "case",
    },
    ]
    experiment_config = {"model": AvailableClassifiers.XGBoost,
                         "test_split": 0.25,
                         "feature_selection": [
                             # {"method": StatisticalFeatureSelect.anova_f,
                             #                    "reduce_to": 8000},
                                               {"method": ModelFeatureSelector.boruta,
                                                "reduce_to": None}
                                               ], }

    exp.add_data_config(data_config, interpreter_config)
    exp.add_model_config(experiment_config)
    exp.run()
