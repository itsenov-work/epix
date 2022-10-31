from utils.logger import LoggerMixin


class ModelAPI(LoggerMixin):

    def __init__(self, model):
        super(ModelAPI, self).__init__()
        self.model = model

    def start_training(self):
        if self.model.is_training:
            self.log.e("The model is already in training!")
        else:
            self.model.train()

    def stop_training(self):
        if not self.model.is_training:
            self.log.e("The model has already stopped training!")
        self.model.stop_training = True

    def get_results(self, num_results):
        with self.model.use_lock:
            self.model.get_results(num_results)

    def save_checkpoint(self):
        with self.model.use_lock:
            self.model.store.save_model()

    def switch_data(self):
        pass
