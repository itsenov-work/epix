import os
import random
import tensorflow as tf
from tfx import v1 as tfx
import tensorflow_model_analysis as tfma
from tfx.components import ImportExampleGen

from data.labelled_images import LabelledImageDataset
from models.classifier import Classifier
from utils.dir import Dir
from utils.optimizer_utils import SAM

from absl import flags, app
from typing import List, Optional, Text

FLAGS = flags.FLAGS
flags.DEFINE_integer("base", 3, "Define base:  EfficientNetB_")
flags.DEFINE_string("dataset_name", "drones_poles_metal", "Choose Dataset: ")
# TODO: implement after discussion
flags.DEFINE_bool("load_if_possible", True, "Whether to continue from previous training if possible.")
flags.DEFINE_bool("enable_tuning", True, "Enable hyperparameter tuning or use initially defined parameters.")

flags.DEFINE_integer("batch_size", 16, "Batch size: ")
flags.DEFINE_string("augmenter", "rand", "Augmentations: auto or rand, rand1, rand2, rand3")
flags.DEFINE_integer("n_epochs", 2, "Number of epochs: ")
flags.DEFINE_integer("n_steps", 2, "Number of steps per epoch: ")
flags.DEFINE_string("optimizer", "adam", "Name of optimiser: ")
flags.DEFINE_bool("use_sam", False, "Use Sharpness Awareness: ")
flags.DEFINE_string("lr", "1e-5", "Optimizer learning rate: ")
flags.DEFINE_integer("freeze_percent", 60, "Freeze percent: ")


def get_image_size(base):
    base_dict = {0: (224, 224),
                 1: (240, 240),
                 2: (260, 260),
                 3: (300, 300),
                 4: (380, 380),
                 5: (456, 456),
                 6: (528, 528),
                 7: (600, 600)}
    return base_dict[base]


def _get_model_path(FLAGS):
    project_root = os.path.dirname(os.path.dirname(__file__))
    models_folder = os.path.join(project_root, "resources", "models")
    return os.path.join(models_folder, 'efficientnet', 'base_' + str(FLAGS.base),
                        FLAGS.optimizer, FLAGS.dataset_name)


def _get_model_defining_flags(FLAGS):
    return {"base": FLAGS.base,
            "lr": FLAGS.lr,
            "optimizer": FLAGS.optimizer,
            "freeze_percent": FLAGS.freeze_percent}


def _create_pipeline(
        pipeline_name: Text,
        pipeline_root: Text,
        data_root: Text,
        module_file: Text,
        accuracy_threshold: float,
        serving_model_dir: Text,
        metadata_path: Text,
        user_provided_schema_path: Optional[Text],
        enable_tuning: bool,
        job_flags: FLAGS,
        enable_bulk_inferrer: bool,
        resolver_range_config: Optional[tfx.proto.RangeConfig],
        beam_pipeline_args: List[Text]
) -> tfx.dsl.Pipeline:
    """Pinched from TFX penguin example:
        https://github.com/tensorflow/tfx/blob/master/tfx/examples/penguin/penguin_pipeline_local.py.
    Args:
    pipeline_name: name of the TFX pipeline being created.
    pipeline_root: root directory of the pipeline.
    data_root: directory containing the penguin data.
    module_file: path to files used in Trainer and Transform components.
    accuracy_threshold: minimum accuracy to push the model.
    serving_model_dir: filepath to write pipeline SavedModel to.
    metadata_path: path to local pipeline ML Metadata store.
    user_provided_schema_path: path to user provided schema file.
    enable_tuning: If True, the hyperparameter tuning through KerasTuner is
      enabled.
    enable_bulk_inferrer: If True, the generated model will be used for a
      batch inference.
    examplegen_input_config: ExampleGen's input_config.
    examplegen_range_config: ExampleGen's range_config.
    resolver_range_config: SpansResolver's range_config. Specify this will
      enable SpansResolver to get a window of ExampleGen's output Spans for
      transform and training.
    beam_pipeline_args: list of beam pipeline options for LocalDAGRunner. Please
      refer to https://beam.apache.org/documentation/runners/direct/.
    Returns:
    A TFX pipeline object.
    """

    # Components:
    schema_importer = None
    schema_gen = None
    examples_resolver = None
    tuner = None
    example_gen_unlabelled = None
    bulk_inferrer = None

    # Brings data into the pipeline or otherwise joins/converts training data.
    example_gen = ImportExampleGen(input_base=data_root)

    # Computes statistics over data for visualization and example validation.
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs['examples'])

    if user_provided_schema_path:
        # Import user-provided schema.
        schema_importer = tfx.dsl.Importer(
            source_uri=user_provided_schema_path,
            artifact_type=tfx.types.standard_artifacts.Schema).with_id(
            'schema_importer')
        schema = schema_importer.outputs['result']
    else:
        # Generates schema based on statistics files.
        schema_gen = tfx.components.SchemaGen(
            statistics=statistics_gen.outputs['statistics'],
            infer_feature_shape=True)
        schema = schema_gen.outputs['schema']

    # Performs anomaly detection based on statistics and data schema.
    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs['statistics'], schema=schema)

    # Gets multiple Spans for transform and training.
    if resolver_range_config:
        examples_resolver = tfx.dsl.Resolver(
            strategy_class=tfx.dsl.experimental.SpanRangeStrategy,
            config={
                'range_config': resolver_range_config
            },
            examples=tfx.dsl.Channel(
                type=tfx.types.standard_artifacts.Examples,
                producer_component_id=example_gen.id)).with_id('span_resolver')

    # Performs transformations and feature engineering in training and serving.
    transform = tfx.components.Transform(
        examples=(examples_resolver.outputs['examples']
                  if resolver_range_config else example_gen.outputs['examples']),
        schema=schema,
        module_file=module_file,
        custom_config={"augmenter": job_flags.augmenter})

    # Tunes the hyperparameters for model training based on user-provided Python
    # function. Note that once the hyperparameters are tuned, you can drop the
    # Tuner component from pipeline and feed Trainer with tuned hyperparameters.
    if enable_tuning:
        tuner = tfx.components.Tuner(
            module_file=module_file,
            examples=transform.outputs['transformed_examples'],
            transform_graph=transform.outputs['transform_graph'],
            train_args=tfx.proto.TrainArgs(num_steps=20),
            eval_args=tfx.proto.EvalArgs(num_steps=5),
            custom_config=_get_model_defining_flags(job_flags))

    # Uses user-provided Python function that trains a model.
    trainer = tfx.components.Trainer(
        module_file=module_file,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema,
        hyperparameters=(tuner.outputs['best_hyperparameters']
                         if enable_tuning else None),
        train_args=tfx.proto.TrainArgs(num_steps=100),
        eval_args=tfx.proto.EvalArgs(num_steps=5),
        custom_config={"model_flags": _get_model_defining_flags(job_flags),
                       "batch_size": job_flags.batch_size})

    # Get the latest blessed model for model validation.
    model_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=tfx.dsl.Channel(
            type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
        'latest_blessed_model_resolver')

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='label')],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(
                    class_name='SparseCategoricalAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': accuracy_threshold}),
                        # Change threshold will be ignored if there is no
                        # baseline model resolved from MLMD (first run).
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': -1e-10})))
            ])
        ])
    evaluator = tfx.components.Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)

    # Checks whether the model passed the validation steps and pushes the model
    # to a file destination if check passed.
    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir)))

    # Showcase for BulkInferrer component.
    if enable_bulk_inferrer:
        # Generates unlabelled examples.
        example_gen_unlabelled = tfx.components.CsvExampleGen(
            input_base=os.path.join(data_root, 'unlabelled')).with_id(
            'CsvExampleGen_Unlabelled')

        # Performs offline batch inference.
        bulk_inferrer = tfx.components.BulkInferrer(
            examples=example_gen_unlabelled.outputs['examples'],
            model=trainer.outputs['model'],
            # Empty data_spec.example_splits will result in using all splits.
            data_spec=tfx.proto.DataSpec(),
            model_spec=tfx.proto.ModelSpec())

    components_list = [
        example_gen,
        statistics_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    ]
    if user_provided_schema_path:
        components_list.append(schema_importer)
    else:
        components_list.append(schema_gen)
    if resolver_range_config:
        components_list.append(examples_resolver)
    if enable_tuning:
        components_list.append(tuner)
    if enable_bulk_inferrer:
        components_list.append(example_gen_unlabelled)
        components_list.append(bulk_inferrer)

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components_list,
        enable_cache=True,
        metadata_connection_config=tfx.orchestration.metadata
            .sqlite_metadata_connection_config(metadata_path),
        beam_pipeline_args=beam_pipeline_args)


def main(unused_argv):
    _pipeline_name = f'drones_tfx'
    project_root = Dir().get_project_dir()
    _module_file = os.path.join(project_root, "src", "tfx_components", "classification_tfx_module.py")
    _serving_model_dir = _get_model_path(FLAGS)
    _tfx_root = os.path.join(project_root, "resources", "tfx")
    # Pipeline root for artifacts.
    _pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
    os.makedirs(_pipeline_root, exist_ok=True)
    # Sqlite ML-metadata db path.
    _metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                                  'metadata.db')

    DATASET_ROOT = r"D:\projects\dataset\1.0.0"
    _beam_pipeline_args = [
        '--direct_running_mode=multi_processing',
        # 0 means auto-detect based on on the number of CPUs available
        # during execution time.
        '--direct_num_workers=1',
    ]
    tfx.orchestration.LocalDagRunner().run(
        _create_pipeline(
            pipeline_name=_pipeline_name,
            pipeline_root=_pipeline_root,
            data_root=DATASET_ROOT,
            module_file=_module_file,
            accuracy_threshold=0.6,
            serving_model_dir=_serving_model_dir,
            metadata_path=_metadata_path,
            user_provided_schema_path=None,
            enable_tuning=FLAGS.enable_tuning,
            enable_bulk_inferrer=False,
            job_flags=FLAGS,
            resolver_range_config=None,
            beam_pipeline_args=_beam_pipeline_args))


if __name__ == '__main__':
    app.run(main)
