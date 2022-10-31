import os
import kerastuner

import tensorflow_transform as tft

from tfx import v1 as tfx
import tensorflow as tf
from absl import flags, app
from tfx_components import model_builders
from tfx_components.inputs_tfx import input_fn


def _get_image_size(base):
    base_dict = {0: (224, 224),
                 1: (240, 240),
                 2: (260, 260),
                 3: (300, 300),
                 4: (380, 380),
                 5: (456, 456),
                 6: (528, 528),
                 7: (600, 600)}
    return base_dict[base]


def tuner_fn(fn_args: tfx.components.FnArgs) -> tfx.components.TunerFnResult:
    """Build the tuner using the KerasTuner API.
    Args:
    fn_args: Holds args as name/value pairs.
      - working_dir: working dir for tuning.
      - train_files: List of file paths containing training tf.Example data.
      - eval_files: List of file paths containing eval tf.Example data.
      - train_steps: number of train steps.
      - eval_steps: number of eval steps.
      - schema_path: optional schema of the input data.
      - transform_graph_path: optional transform graph produced by TFT.
    Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.

    NOTE FROM MATYO: this is very heavily based on the penguin example:
        https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_utils_keras.py
    """
    transform_graph = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_dataset = input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        transform_graph,
        fn_args.custom_config.get("batch_size"))

    eval_dataset = input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        transform_graph,
        fn_args.custom_config.get("batch_size"))
    tuning_builder = model_builders.TunedModel(flags=fn_args.custom_config.get("model_flags"),
                                               n_classes=train_dataset.num_classes,
                                               model_root=fn_args.model_run_dir)
    tuner = kerastuner.RandomSearch(
        tuning_builder.build,
        max_trials=6,
        hyperparameters=tuning_builder.get_hyperparameters(),
        allow_new_entries=False,
        objective=kerastuner.Objective('val_sparse_categorical_accuracy', 'max'),
        directory=fn_args.working_dir,
        project_name='drones_tuning')

    return tfx.components.TunerFnResult(
        tuner=tuner,
        fit_kwargs={
          'x': train_dataset,
          'validation_data': eval_dataset,
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps
        })


def run_fn(fn_args: tfx.components.FnArgs):
    """Train the model based on given args.
    Args:
    fn_args: Holds args used to train the model as name/value pairs.
    """

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        fn_args.custom_config.get("batch_size"))

    eval_dataset = input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        fn_args.custom_config.get("batch_size"))

    builder = model_builders.FixedModel(flags=fn_args.custom_config.get("model_flags"),
                                        n_classes=train_dataset.num_classes,
                                        model_root=fn_args.model_run_dir)

    if fn_args.hyperparameters:
        hparams = kerastuner.HyperParameters.from_config(fn_args.hyperparameters)
    else:
        hparams = builder.get_hyperparameters()

    model = builder.build(hparams)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq='batch')
    savedmodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath=fn_args.model_run_dir, save_best_only=True)
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback,
                   savedmodel_callback])