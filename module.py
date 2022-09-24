import tensorflow as tf
import tensorflow_transform as tft 
import os 


NUMERIC_FEATURE_KEYS = [
    'age', 'capital-gain', 'capital-loss',
    'education-num', 'fnlwgt', 'hours-per-week'
]

ONE_HOT_FEATURES = {
    'education':16, 'marital-status':7, 'native-country':41,
    'occupation':15, 'race':5, 'relationship':6, 'sex':2,
    'workclass':9
}


NUM_OOV_BUCKETS = 2

LABEL_KEY = 'label'

def transformed_name(key: str) -> str:
    key = key.replace('-','_')
    return key + '_xf'

def preprocessing_fn(inputs):

    outputs = {}

    for key in NUMERIC_FEATURE_KEYS:
        scaled = tft.scale_to_0_1(inputs[key])
        outputs[transformed_name(key)] = tf.reshape(scaled, [-1])

    for key, size in ONE_HOT_FEATURES.items():
        indices = tft.compute_and_apply_vocabulary(inputs[key],num_oov_buckets = NUM_OOV_BUCKETS ) 
        one_hot = tf.one_hot(indices, size + NUM_OOV_BUCKETS)   
        outputs[transformed_name(key)] = tf.reshape(one_hot, [-1, size + NUM_OOV_BUCKETS])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.float32)
    return outputs

def get_model(show_summary: bool = True) -> tf.keras.models.Model:
  """
  This function defines a Keras model and returns the model as a keras object.
  
  """

  #one hot categorical features 
  one_hot_features = []
  for keys,  dim in ONE_HOT_FEATURES.items():
    one_hot_features.append(
      tf.keras.Input(shape=(dim+NUM_OOV_BUCKETS,), name=transformed_name(keys),dtype=tf.float32)
    )

  #numerical features
  numeric_features = []
  for keys in NUMERIC_FEATURE_KEYS:
    numeric_features.append(
      tf.keras.Input(shape=(1,), dtype=tf.float32, name=transformed_name(keys))
    )

  input_layers = one_hot_features + numeric_features

  input_numeric = tf.keras.layers.concatenate(numeric_features)
  input_categorical = tf.keras.layers.concatenate(one_hot_features)

  concat = tf.keras.layers.concatenate([input_numeric, input_categorical])
  deep = tf.keras.layers.Dense(256, activation='relu')(concat)
  deep = tf.keras.layers.Dense(64, activation='relu')(deep)
  deep = tf.keras.layers.Dense(16, activation='relu')(deep)
  output = tf.keras.layers.Dense(1, activation='sigmoid')(deep)

  keras_model = tf.keras.models.Model(input_layers, output)
  keras_model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
      tf.keras.metrics.BinaryAccuracy(),
      tf.keras.metrics.TruePositives(),
    ],
  )
  keras_model.summary()
  return keras_model

def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')



def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn  


def _input_fn(file_pattern, tf_transform_output, batch_size=64):
    """Generates features and label for tuning/training.
    Args:
    file_pattern: input tfrecord file pattern.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch
      Returns:
        A dataset that contains (features, indices) tuple where features is a
          dictionary of Tensors, and indices is a single Tensor of
          label indices.
    """
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset

def run_fn(fn_args):
    """Train the model based on given args.
    Args:
    fn_args: Holds args used to train the model as name/value pairs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 64)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 64)

    model = get_model()

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )
    callbacks = [tensorboard_callback]

    model.fit(
        train_dataset,
        epochs=1,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=callbacks,
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }
    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)



    
    
 

