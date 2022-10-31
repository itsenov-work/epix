import kerastuner as kt


class OurTuner(kt.Tuner):
    def __init__(self, trainer, *args, **kwargs):
        super(OurTuner, self).__init__(*args, **kwargs)
        self.trainer = trainer

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        model = self.hypermodel.build(trial.hyperparameters)
