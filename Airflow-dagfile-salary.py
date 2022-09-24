from tfx import components
import tensorflow_model_analysis as tfma 
from tfx.components import (CsvExampleGen, Evaluator, ExampleValidator, Pusher, SchemaGen, StatisticsGen, Trainer, Transform)
from tfx.proto import pusher_pb2, trainer_pb2
from tfx.proto import example_gen_pb2
from tfx import v1 as tfx
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
import os
import absl
from datetime import datetime
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig


airflow_config = {
    "schedule_interval": None,
    "start_date": datetime(2022, 1, 1),
}

pipeline_dir = '/home/dinuka/airflow/tfx_101'
pipeline_name = 'salary_pipeline'
data_dir =  os.path.join(pipeline_dir, 'data')
module_file = os.path.join(pipeline_dir, 'components', 'module.py')
output_base = os.path.join(pipeline_dir, 'output', pipeline_name)
serving_model_dir = os.path.join(output_base, pipeline_name)
pipeline_root = os.path.join(output_base, 'pipeline_root')
metadata_path = os.path.join(pipeline_root, 'metadata.sqlite')



def init_components(data_root, module_file, serving_model_dir,training_steps=2000, eval_steps=200):


    output = example_gen_pb2.Output(split_config = example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=6),
            example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2),
            example_gen_pb2.SplitConfig.Split(name='test', hash_buckets=2)
        ])
  )
        
    example_gen = CsvExampleGen(input_base = data_root, output_config =output)

    statistics_gen = StatisticsGen(examples = example_gen.outputs['examples'])

    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)

    example_validator = ExampleValidator(statistics = statistics_gen.outputs['statistics'],schema=schema_gen.outputs['schema'])

    transform = Transform(examples=example_gen.outputs['examples'],schema=schema_gen.outputs['schema'],module_file=module_file)

    trainer = Trainer(module_file=(module_file),transformed_examples=transform.outputs['transformed_examples'],transform_graph=transform.outputs['transform_graph'],
                        schema=schema_gen.outputs['schema'],train_args=trainer_pb2.TrainArgs(num_steps=training_steps),eval_args=trainer_pb2.EvalArgs(num_steps=eval_steps))

    model_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=tfx.dsl.Channel(type=tfx.types.standard_artifacts.ModelBlessing),
    ) 
    
    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(
                signature_name="serving_default",
                label_key="label",
            )
        ],
        slicing_specs=[tfma.SlicingSpec(), tfma.SlicingSpec(feature_keys=["product"])],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(
                        class_name="BinaryAccuracy",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": 0.65}
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": -1e-10},
                            ),
                        ),
                    ),
                    tfma.MetricConfig(class_name="Precision"),
                    tfma.MetricConfig(class_name="Recall"),
                    tfma.MetricConfig(class_name="ExampleCount"),
                    tfma.MetricConfig(class_name="AUC"),
                ],
            )
        ],
    )

    evaluator = Evaluator(examples=example_gen.outputs['examples'],model=trainer.outputs['model'],eval_config=eval_config)

    pusher = Pusher(model=trainer.outputs['model'],model_blessing=evaluator.outputs['blessing'],
                    push_destination=pusher_pb2.PushDestination(filesystem=pusher_pb2.PushDestination.Filesystem(base_directory=serving_model_dir)))

    
    
    components=[
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator,
            transform,
            trainer,
            evaluator,
            pusher,
        ]

    return components
                        
def init_beam_pipeline(components, pipeline_root, direct_num_workers):

  absl.logging.info("Pipeline root set to:{}".format(pipeline_root))
  beam_arg =[
      "--direct_num_workers={}".format(direct_num_workers),
  ]

  p = pipeline.Pipeline(
      pipeline_name = pipeline_name,
      pipeline_root = pipeline_root,
      components = components,
      enable_cache = False,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path),
      beam_pipeline_args=beam_arg
  )
  return p

components = init_components(data_dir, module_file, serving_model_dir,
 training_steps=3000, eval_steps=2000)
pipeline = init_beam_pipeline(components, pipeline_root, 2)
DAG = AirflowDagRunner(AirflowPipelineConfig(airflow_config)).run(pipeline)                          
