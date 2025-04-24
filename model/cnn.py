from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)  # Use tf.compat.v1 for TF 2.x compatibility


def cnn_model_fn(features, labels, mode):
  """Model function for a basic CNN.

  Args:
    features: Input features (e.g., images).
    labels: Labels for training or evaluation.
    mode: Indicates if we're in TRAIN, EVAL, or PREDICT mode.

  Returns:
    tf.estimator.EstimatorSpec: Specification for the estimator.
  """

  # Input Layer
  # Reshape input to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.compat.v1.layers.conv2d(  # Use tf.compat.v1.layers for TF 2.x
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      name="conv1")  # Added name
  tf.compat.v1.summary.histogram("conv1_out", conv1)  # Added summary

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.compat.v1.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name="pool1")  # Use tf.compat.v1.layers
  tf.compat.v1.summary.histogram("pool1_out", pool1)  # Added summary

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.compat.v1.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      name="conv2")
  tf.compat.v1.summary.histogram("conv2_out", conv2)  # Added summary

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.compat.v1.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name="pool2")
  tf.compat.v1.summary.histogram("pool2_out", pool2)  # Added summary

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.compat.v1.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name="dense1")
  tf.compat.v1.summary.histogram("dense1_out", dense)  # Added summary

  # Add dropout operation; 0.4 probability that element will be dropped (0.6 kept)
  dropout = tf.compat.v1.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN, name="dropout")
  tf.compat.v1.summary.histogram("dropout_out", dropout)  # Added summary

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.compat.v1.layers.dense(inputs=dropout, units=10, name="logits")
  tf.compat.v1.summary.histogram("logits_out", logits)  # Added summary

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1, name="class_predictions"),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  tf.compat.v1.summary.scalar("loss", loss)  # Added summary

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)  # Use tf.compat.v1.train
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.compat.v1.train.get_global_step())  # Use tf.compat.v1.train
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.compat.v1.metrics.accuracy(  # Use tf.compat.v1.metrics
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  mnist = tf.compat.v1.keras.datasets.mnist  # Use tf.compat.v1.keras.datasets
  (train_data, train_labels), (eval_data, eval_labels) = mnist.load_data()

  train_data = train_data / 255.0  # Normalize pixel values
  eval_data = eval_data / 255.0

  train_data = np.expand_dims(train_data, axis=-1)  # Add channel dimension
  eval_data = np.expand_dims(eval_data, axis=-1)

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.compat.v1.train.LoggingTensorHook(  # Use tf.compat.v1.train
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(  # Use tf.compat.v1.estimator.inputs
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=1000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x={"x": eval_data}, y=eval_labels, batch_size=100, num_epochs=1, shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.compat.v1.app.run()  # Use tf.compat.v1.app