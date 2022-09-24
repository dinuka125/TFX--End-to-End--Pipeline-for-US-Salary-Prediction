# TFX--End-to-End--Pipeline-for-US-Salary-Prediction
US Salary Prediction Repository Demonstrate end-to-end workflow implementation using TFX 

Following Components were used 

 - ExampleGen - ingests and splits the input dataset.
 - StatisticsGen - calculates statistics for the dataset.
 - SchemaGen - examines the statistics and creates a data schema.
 - ExampleValidator - looks for anomalies and missing values in the dataset.
 - Transform performs - feature engineering on the dataset.
 - Trainer - trains the model using TensorFlow Estimators or Keras.
 - Evaluator - performs deep analysis of the training results.
 - Pusher - deploys the model to a serving infrastructure.

Tensorflow serving is used for the deployement
