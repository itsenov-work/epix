import tensorflow as tf
from framework.model import Model
import pickle
from tensorflow.python.lib.io import file_io


class WeightCheckpoint(tf.train.Checkpoint):
    def __init__(self, step_counter, model: Model):
        super(WeightCheckpoint, self).__init__()
        self.model = model
        self.step_counter = step_counter

    def write(self, file_prefix, options=None):
        weight_dict = dict()
        for submodel_name, submodel in self.model.submodels.items():
            weight_dict[submodel_name] = submodel.get_weights()

        pickle_name = "%s-%d" % (file_prefix, self.step_counter)
        with open(pickle_name, 'wb') as file:
            pickle.dump(weight_dict, file)

        return pickle_name

    def read(self, save_path, options=None):
        with open(save_path, 'rb') as file:
            weight_dict = pickle.load(file)
        for submodel_name, submodel in self.model.submodels.items():
            submodel.set_weights(weight_dict[submodel_name])


class WeightCheckpointManager(tf.train.CheckpointManager):
    """
        Workaround for the weight files being pickles instead of ckpt files,
        meaning they weren't being deleted properly.
    """

    def _sweep(self):
        if not self._max_to_keep:
            return
        while len(self._maybe_delete) > self._max_to_keep:
            filename, timestamp = self._maybe_delete.popitem(last=False)
            if (self._keep_checkpoint_every_n_hours
                    and (timestamp - self._keep_checkpoint_every_n_hours * 3600.
                         >= self._last_preserved_timestamp)):
                self._last_preserved_timestamp = timestamp
                continue
            self._delete_file_if_exists(filename)

    @staticmethod
    def _delete_file_if_exists(filespec):
        """Deletes files matching `filespec`."""
        for pathname in file_io.get_matching_files(filespec):
            file_io.delete_file(pathname)


class SavedModelCheckpoint(tf.train.Checkpoint):
    def __init__(self, step_counter, model: Model, spec):
        super(SavedModelCheckpoint, self).__init__()
        self.model = model
        self.step_counter = step_counter
        self.spec = spec

    def write(self, file_prefix, options=None):
        model_name = "%s-%d" % (file_prefix, self.step_counter)
        tf.saved_model.save(self.model, model_name)

        return model_name

    def read(self, save_path, options=None):
        self.model = tf.saved_model.load(save_path)

    def restore(self, *args):
        self.read(*args)
