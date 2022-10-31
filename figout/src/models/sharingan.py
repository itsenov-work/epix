from typing import List

from _legacy.data.data_provider import DataProvider
from framework.model import Model, ModelResults


class SharinGAN(Model):
    def __init__(self, models: List[Model]):
        super(SharinGAN, self).__init__()
        self.models = models

    def compile(self, *args, **kwargs):
        for model in self.models:
            model.compile(*args)

    def train_step(self, data: List[DataProvider]):
        for m, d in zip(self.models, data):
            m.train_step(d)

    def get_results(self, num_outputs):
        results = ModelResults(data=[], names=[])
        for model in self.models:
            results.add_results(model.get_results(num_outputs))
        return results
