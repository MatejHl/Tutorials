import os
import tensorflow as tf

import apache_beam

import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam

import requests
import tempfile

train = './data/adult.data'
test = './data/adult.test'

# DOWNLOAD DATA: -----------------------
# train_data_url = 'https://storage.googleapis.com/artifacts.tfx-oss-public.appspot.com/datasets/census/adult.data'
# test_data_url = 'https://storage.googleapis.com/artifacts.tfx-oss-public.appspot.com/datasets/census/adult.test'

# with open(train, 'wb') as file:
#     response = requests.get(train_data_url, allow_redirects=False)
#     file.write(response.content)
# 
# with open(test, 'wb') as file:
#     response = requests.get(test_data_url, allow_redirects=False)
#     file.write(response.content)
# ----------------------- -----------------------

CATEGORICAL_FEATURE_KEYS = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
]
NUMERIC_FEATURE_KEYS = [
    'age',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
]
OPTIONAL_NUMERIC_FEATURE_KEYS = [
    'education-num',
]
LABEL_KEY = 'label'

# MetaData:
RAW_DATA_FEATURE_SPEC = dict(
    [(name, tf.io.FixedLenFeature([], tf.string))
     for name in CATEGORICAL_FEATURE_KEYS] +
    [(name, tf.io.FixedLenFeature([], tf.float32))
     for name in NUMERIC_FEATURE_KEYS] +
    [(name, tf.io.VarLenFeature(tf.float32))
     for name in OPTIONAL_NUMERIC_FEATURE_KEYS] +
    [(LABEL_KEY, tf.io.FixedLenFeature([], tf.string))]
)

RAW_DATA_METADATA = tft.tf_metadata.dataset_metadata.DatasetMetadata(
    tft.tf_metadata.dataset_schema.schema_utils.schema_from_feature_spec(RAW_DATA_FEATURE_SPEC))



testing = os.getenv("WEB_TEST_BROWSER", False)
if testing:
  TRAIN_NUM_EPOCHS = 1
  NUM_TRAIN_INSTANCES = 1
  TRAIN_BATCH_SIZE = 1
  NUM_TEST_INSTANCES = 1
else:
  TRAIN_NUM_EPOCHS = 16
  NUM_TRAIN_INSTANCES = 32561
  TRAIN_BATCH_SIZE = 128
  NUM_TEST_INSTANCES = 16281


# Names of temp files
TRANSFORMED_TRAIN_DATA_FILEBASE = 'train_transformed'
TRANSFORMED_TEST_DATA_FILEBASE = 'test_transformed'
EXPORTED_MODEL_DIR = 'exported_model_dir'





class MapAndFilterErrors(apache_beam.PTransform):
  """Like beam.Map but filters out errors in the map_fn."""

  class _MapAndFilterErrorsDoFn(apache_beam.DoFn):
    """Count the bad examples using a beam metric."""

    def __init__(self, fn):
      self._fn = fn
      # Create a counter to measure number of bad elements.
      self._bad_elements_counter = apache_beam.metrics.Metrics.counter(
          'census_example', 'bad_elements')

    def process(self, element):
      try:
        yield self._fn(element)
      except Exception:  # pylint: disable=broad-except
        # Catch any exception the above call.
        self._bad_elements_counter.inc(1)

  def __init__(self, fn):
    self._fn = fn

  def expand(self, pcoll):
    return pcoll | apache_beam.ParDo(self._MapAndFilterErrorsDoFn(self._fn))







def preprocessing_fn(inputs):
  """
  Preprocess input columns into transformed columns.
  """
  # Since we are modifying some features and leaving others unchanged, we
  # start by setting `outputs` to a copy of `inputs.
  outputs = inputs.copy()

  # Scale numeric columns to have range [0, 1].
  for key in NUMERIC_FEATURE_KEYS:
    outputs[key] = tft.scale_to_0_1(outputs[key])

  for key in OPTIONAL_NUMERIC_FEATURE_KEYS:
    # This is a SparseTensor because it is optional. Here we fill in a default
    # value when it is missing.
    sparse = tf.sparse.SparseTensor(outputs[key].indices, outputs[key].values,
                                    [outputs[key].dense_shape[0], 1])
    dense = tf.sparse.to_dense(sp_input=sparse, default_value=0.)
    # Reshaping from a batch of vectors of size 1 to a batch to scalars.
    dense = tf.squeeze(dense, axis=1)
    outputs[key] = tft.scale_to_0_1(dense)

    # For all categorical columns except the label column, we generate a
    # vocabulary but do not modify the feature.  This vocabulary is instead
    # used in the trainer, by means of a feature column, to convert the feature
    # from a string to an integer id.
    for key in CATEGORICAL_FEATURE_KEYS:
      tft.vocabulary(inputs[key], vocab_filename=key)

    # For the label column we provide the mapping from string to index.
    table_keys = ['>50K', '<=50K']
    initializer = tf.lookup.KeyValueTensorInitializer(
        keys=table_keys,
        values=tf.cast(tf.range(len(table_keys)), tf.int64),
        key_dtype=tf.string,
        value_dtype=tf.int64)
    table = tf.lookup.StaticHashTable(initializer, default_value=-1)
    outputs[LABEL_KEY] = table.lookup(outputs[LABEL_KEY])

  return outputs


# Apache Beam syntax: result = pass_this | 'name this step' >> to_this_call


def transform_data(train_data_file, test_data_file, working_dir):
  """Transform the data and write out as a TFRecord of Example protos.

  Read in the data using the CSV reader, and transform it using a
  preprocessing pipeline that scales numeric data and converts categorical data
  from strings to int64 values indices, by creating a vocabulary for each
  category.

  Args:
    train_data_file: File containing training data
    test_data_file: File containing test data
    working_dir: Directory to write transformed data and metadata to
  """

  # The "with" block will create a pipeline, and run that pipeline at the exit
  # of the block.
  with apache_beam.Pipeline() as pipeline:
    with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
      # Create a coder to read the census data with the schema.  To do this we
      # need to list all columns in order since the schema doesn't specify the
      # order of columns in the csv.
      ordered_columns = [
          'age', 'workclass', 'fnlwgt', 'education', 'education-num',
          'marital-status', 'occupation', 'relationship', 'race', 'sex',
          'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
          'label'
      ]
      converter = tft.coders.CsvCoder(ordered_columns, RAW_DATA_METADATA.schema)

      # Read in raw data and convert using CSV converter.  Note that we apply
      # some Beam transformations here, which will not be encoded in the TF
      # graph since we don't do them from within tf.Transform's methods
      # (AnalyzeDataset, TransformDataset etc.).  These transformations are just
      # to get data into a format that the CSV converter can read, in particular
      # removing spaces after commas.
      #
      # We use MapAndFilterErrors instead of Map to filter out decode errors in
      # convert.decode which should only occur for the trailing blank line.
      raw_data = (
          pipeline
          | 'ReadTrainData' >> apache_beam.io.ReadFromText(train_data_file)
          | 'FixCommasTrainData' >> apache_beam.Map(
              lambda line: line.replace(', ', ','))
          | 'DecodeTrainData' >> MapAndFilterErrors(converter.decode))

      # Combine data and schema into a dataset tuple.  Note that we already used
      # the schema to read the CSV data, but we also need it to interpret
      # raw_data.
      raw_dataset = (raw_data, RAW_DATA_METADATA)
      transformed_dataset, transform_fn = (
          raw_dataset | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
      transformed_data, transformed_metadata = transformed_dataset
      # A coder between TF Examples and tf.Transform datasets.
      # Used to encode a tf.transform encoded dict as tf.Example.
      transformed_data_coder = tft.coders.ExampleProtoCoder(
          transformed_metadata.schema)

      _ = (
          transformed_data
          | 'EncodeTrainData' >> apache_beam.Map(transformed_data_coder.encode)
          | 'WriteTrainData' >> apache_beam.io.WriteToTFRecord(
              os.path.join(working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE)))

      # Now apply transform function to test data.  In this case we remove the
      # trailing period at the end of each line, and also ignore the header line
      # that is present in the test data file.
      raw_test_data = (
          pipeline
          | 'ReadTestData' >> apache_beam.io.ReadFromText(test_data_file,
                                                   skip_header_lines=1)
          | 'FixCommasTestData' >> apache_beam.Map(
              lambda line: line.replace(', ', ','))
          | 'RemoveTrailingPeriodsTestData' >> apache_beam.Map(lambda line: line[:-1])
          | 'DecodeTestData' >> MapAndFilterErrors(converter.decode))

      raw_test_dataset = (raw_test_data, RAW_DATA_METADATA)

      transformed_test_dataset = (
          (raw_test_dataset, transform_fn) | tft_beam.TransformDataset())
      # Don't need transformed data schema, it's the same as before.
      transformed_test_data, _ = transformed_test_dataset

      _ = (
          transformed_test_data
          | 'EncodeTestData' >> apache_beam.Map(transformed_data_coder.encode)
          | 'WriteTestData' >> apache_beam.io.WriteToTFRecord(
              os.path.join(working_dir, TRANSFORMED_TEST_DATA_FILEBASE)))

      # Will write a SavedModel and metadata to working_dir, which can then
      # be read by the tft.TFTransformOutput class.
      _ = (
          transform_fn
          | 'WriteTransformFn' >> tft_beam.WriteTransformFn(working_dir))




def _make_training_input_fn(tf_transform_output, transformed_examples,
                            batch_size):
  """Creates an input function reading from transformed data.

  Args:
    tf_transform_output: Wrapper around output of tf.Transform.
    transformed_examples: Base filename of examples.
    batch_size: Batch size.

  Returns:
    The input function for training or eval.
  """
  def input_fn():
    """
    Input function for training and eval.
    
    tf_transform_output is used to get specifications from transformations made. It gets them from the directory containing tf_transform output (working_dir).
    """
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=transformed_examples,
        batch_size=batch_size,
        features=tf_transform_output.transformed_feature_spec(),
        reader=tf.data.TFRecordDataset,
        shuffle=True)

    transformed_features = tf.compat.v1.data.make_one_shot_iterator(
        dataset).get_next()

    # Extract features and label from the transformed tensors.
    transformed_labels = transformed_features.pop(LABEL_KEY)

    return transformed_features, transformed_labels

  return input_fn




working_dir = 'Transformed_data'

# transform_data(train, test, working_dir)

tf_transform_output = tft.TFTransformOutput(working_dir)
input_fn = _make_training_input_fn(tf_transform_output, 
                        transformed_examples = os.path.join(working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE + '*'),
                            batch_size = TRAIN_BATCH_SIZE)

