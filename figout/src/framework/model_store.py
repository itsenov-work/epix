from utils.logger import LoggerMixin
import os
from skimage import io
from utils.google_storage import GoogleStorage
from utils.misc import MiscUtils
import json


class ModelStore(LoggerMixin):

    def __init__(self, model):
        super(ModelStore, self).__init__()
        self.model = model

        # Generic files system for models
        self._dirs = ModelFiles(self)
        # Model results
        self._results = ModelResults(self)
        # Model arguments
        self._arguments = ModelArguments(self)
        # Submodel connection
        # self.sm_manager = SubmodelStoreManager(self)

    def results(self):
        return self._results

    def dirs(self):
        return self._dirs

    def arguments(self):
        return self._arguments.read()

    def save_results(self, results):
        self.results().store(results)

    def send_results(self):
        self.results().send()

    def clear_results(self):
        self.results().clear()


class SubmodelStoreManager(LoggerMixin):
    def __init__(self, store: ModelStore):
        super().__init__()
        self.substores = self._extract_submodel_savers(store.model)
        self.parent_dir = store.dirs()

    @staticmethod
    def _extract_submodel_savers(model):
        submodels = model.submodels
        return [submodel.store for submodel in submodels]

    def save_submodels(self):
        for store in self.substores:
            store.save_model()


def models_folder():
    return os.path.join("resources", "models")


class ModelDirectoryManager(LoggerMixin):
    subdir = None

    def __init__(self, store: ModelStore):
        super(ModelDirectoryManager, self).__init__()
        self.store = store
        self.dir = os.path.join(self.store.dirs().model_dir, self.subdir)
        os.makedirs(self.dir, exist_ok=True)

    def clear(self):
        MiscUtils().clear_folder(self.dir)

    def send(self):
        gsw = GoogleStorage()
        gsw.upload(self.dir)


class ModelResults(ModelDirectoryManager):
    subdir = "results"

    def store(self, results):
        from utils.misc import MiscUtils
        MiscUtils().clear_folder(self.dir)
        for i, result in enumerate(results):
            filename = 'results{}.png'.format(i)
            image_path = os.path.join(self.dir, filename)
            from skimage import img_as_ubyte
            io.imsave(image_path, img_as_ubyte(result), check_contrast=False)


class ModelFiles:
    def __init__(self, store: ModelStore):
        self.model = store.model
        self.model_class_dir = self._model_class_dir()
        os.makedirs(self.model_class_dir, exist_ok=True)
        self.model_dir = self._model_dir()
        os.makedirs(self.model_dir, exist_ok=True)

    def _model_class_dir(self):
        # Ex: resources/models/GAN
        return os.path.join(models_folder(), self.model.__class__.__name__)

    def _model_dir(self):
        # Ex: resources/models/GAN/batch_size16_size_256x256_0
        subdirs = sorted(os.listdir(self.model_class_dir))
        subdirs = [s for s in subdirs if os.path.isdir(os.path.join(self.model_class_dir, s))]
        runtime_name = self.model.state.runtime_name
        indices = [int(subdir.split("_")[-1]) for subdir in subdirs if subdir.startswith(runtime_name)]
        index = max(indices) if len(indices) > 0 else 0
        index = index + 1
        model_dir_name = "%s_%d" % (runtime_name, index)
        return os.path.join(self.model_class_dir, model_dir_name)


class ModelArguments:
    def __init__(self, store: ModelStore):
        self.model = store.model
        self.model_dir = store.dirs().model_dir
        self.file_path = os.path.join(self.model_dir, 'arguments.json')

        with open(self.file_path, 'w+') as f:
            json.dump(self.model.arguments, f, indent=4)

    def read(self):
        with open(self.file_path, 'r') as f:
            return json.load(f)
