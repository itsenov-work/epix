import yaml
from utils.easydict import EasyDict
from utils.logger import LoggerAttachMixin


class YamlConfigBuilder(LoggerAttachMixin):
    REQUIRED_FIELDS = []
    OPTIONAL_FIELDS = {}
    _yaml = None

    def load_yaml(self, filename):
        if self.log is None:
            self.attach_log()
        if not filename.endswith(".yaml"):
            filename += ".yaml"

        self.log.i(f"Loading yaml file {filename}")
        with open(filename, 'r') as f:
            yml = yaml.load(f, Loader=yaml.SafeLoader)
        self._yaml = EasyDict(yml)
        errors = []
        for f in self.REQUIRED_FIELDS:
            if f not in self._yaml:
                errors.append(f)
        if len(errors) > 0:
            self.log.e(f"YAML load failed, missing required fields: {errors}")

    def getprop(self, field):
        if not field in set(self.REQUIRED_FIELDS).union(self.OPTIONAL_FIELDS.keys()):
            self.log.e(f"Field {field} not found in yaml configuration")
        if field in self.REQUIRED_FIELDS:
            return self._yaml.get(field)
        return self._yaml.get(field, self.OPTIONAL_FIELDS[field])

