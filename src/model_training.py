from typing import Dict

import catboost
import numpy as np
import pandas as pd
import xgboost as xgb
from boruta import BorutaPy
from catboost import CatBoostClassifier
from scipy.stats import chi2
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, classification_report
from strenum import StrEnum

from utils.logging import LoggerMixin


class AvailableClassifiers(StrEnum):
    XGBoost = 'xgboost'
    CatBoost = 'catboost'


class StatisticalFeatureSelect(StrEnum):
    anova_f = "ANOVA"
    chi_squared = "chi_squared"


class ModelFeatureSelector(StrEnum):
    boruta = "boruta"


class ModelTrainer(LoggerMixin):
    def __init__(self):
        super(ModelTrainer, self).__init__()
        self.model = self._define_model()
        self.model_type = None
        self.feature_selectors = []
        self.df = None
        self.training_data: Dict[str, pd.DataFrame] = {"x": pd.DataFrame(),
                                                       "y": pd.DataFrame()}
        self.testing_data: Dict[str, pd.DataFrame] = {"x": pd.DataFrame(),
                                                      "y": pd.DataFrame()}

    def add_dataset(self, data):
        self.training_data = data["training_data"]
        self.testing_data = data["testing_data"]

    @staticmethod
    def _define_model():
        return None

    def add_feature_selection(self, feature_selector: StrEnum or str, reduce_to_n: int = None):
        if feature_selector == StatisticalFeatureSelect.anova_f:
            if reduce_to_n is None:
                self.log.w("Feature selection: the ANOVA selection reduction set to 8000 by default. Please give"
                           "a value to 'reduce_to_n' parameter to change.")
                reduce_to_n = 8000

            self.feature_selectors.append(SelectKBest(f_classif, k=reduce_to_n))
        if feature_selector == StatisticalFeatureSelect.chi_squared:
            if reduce_to_n is None:
                self.log.w("Feature selection: the Chi-Squared selection reduction set to 8000 by default. Please give"
                           "a value to 'reduce_to_n' parameter to change.")
                reduce_to_n = 8000
            self.feature_selectors.append(SelectKBest(chi2, k=reduce_to_n))
        if feature_selector == ModelFeatureSelector.boruta:
            self.feature_selectors.append(BorutaWrapper(
                estimator=self.model,
                n_estimators='auto',
                max_iter=20  # number of trials to perform
            ))

    def feature_select(self):
        initial_n = self.training_data["x"].shape[1]
        for selector in self.feature_selectors:
            selector.fit(self.training_data["x"], self.training_data["y"])
            feature_idx = selector.get_support(indices=True)
            feature_names = self.training_data["x"].columns[feature_idx]
            self.training_data["x"] = pd.DataFrame(selector.transform(self.training_data["x"]),
                                                   index=self.training_data["x"].index, columns=feature_names)
            self.testing_data["x"] = pd.DataFrame(selector.transform(self.testing_data["x"]),
                                                  index=self.testing_data["x"].index, columns=feature_names)
        final_n = self.training_data["x"].shape[1]
        self.log.s("Reduced features from {} to {}.".format(initial_n, final_n))

    def train(self):
        self.model.fit(self.training_data["x"], self.training_data["y"])
        y_pred = self.model.predict(self.testing_data["x"])
        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(self.testing_data["y"], predictions)
        self.log.s("Successfully trained model with accuracy: {}".format(np.round(accuracy * 100.0, 2)))

    def get_model(self):
        return self.model

    def test(self):
        return self.testing_data["y"], self.model.predict(self.testing_data["x"])

    def get_confusion_matrix(self, ax):
        cmd = ConfusionMatrixDisplay(confusion_matrix(self.testing_data["y"],
                                                      self.model.predict(self.testing_data["x"])))
        cmd.plot(ax=ax)

    def get_metrics_report(self):
        return classification_report(self.testing_data["y"], self.model.predict(self.testing_data["x"]),
                                     output_dict=True)

    def get_feature_importance(self, ax):
        pass

    def save_model(self, path):
        self.model.save_model(path)


class XGBTrainer(ModelTrainer):
    @staticmethod
    def _define_model():
        return xgb.XGBClassifier()

    def get_feature_importance(self, ax):
        xgb.plot_importance(self.model, ax=ax, max_num_features=20)


class CatTrainer(ModelTrainer):
    @staticmethod
    def _define_model():
        return catboost.CatBoostClassifier(depth=20)

    def get_feature_importance(self, ax):
        importance = self.model.get_feature_importance()


class BorutaWrapper(BorutaPy):
    def get_support(self, indices=True):
        return self.support_

    def fit(self, X, y):
        super(BorutaWrapper, self).fit(np.array(X), np.array(y))

    def transform(self, X, weak=False):
        return super(BorutaWrapper, self)._transform(np.array(X), weak=weak)
