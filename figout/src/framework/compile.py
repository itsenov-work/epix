from framework.model import Model


class CompileConfig:
    pass


class Compiler:
    def __init__(self, model: Model, config: CompileConfig):
        self.model = model
        self.config = config

    def compile(self):
        self.model.compile(self.config)
