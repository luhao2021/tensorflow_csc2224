from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
import time

# import os
# os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# os.environ['TF_DUMP_GRAPH_PREFIX'] = '/tmp/tf_graphs/'
# os.environ['XLA_FLAGS'] = "--xla_dump_hlo_as_text --xla_dump_to=/tmp/tf_graphs/xla_dump"


num_classes = 10 # 0 to 9 digits
num_features = 784 # 28*28

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Flatten images to 1-D vector of 784 features (28*28).
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.

# Training parameters.
learning_rate = 0.01
training_steps = 1000
display_step = 50
batch_size = 256

train_data=tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_data=train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

W = tf.Variable(tf.ones([num_features, num_classes]), name="weight")
b = tf.Variable(tf.zeros([num_classes]), name="bias")

dtype = tf.cus

@tf.function(experimental_compile=True)
def model(x):
  # Apply softmax to normalize the logits to a probability distribution.
  x_cast = tf.cast(x, dtype)
  w_cast = tf.cast(W, dtype)
  b_cast = tf.cast(b, dtype)
  out = tf.matmul(x_cast, w_cast) + b_cast
  out = tf.cast(out, tf.float32)
  out = tf.nn.softmax(out)
  return out

# Cross-Entropy loss function.
def cross_entropy(y_pred, y_true):
  # Encode label to a one hot vector.
  y_true = tf.one_hot(y_true, depth=num_classes)
  # Clip prediction values to avoid log(0) error.
  y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
  # Compute cross-entropy.
  return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))


def accuracy(y_pred, y_true):
  # Predicted class is the index of the highest score in prediction vector (i.e. argmax).
  correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
  return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(learning_rate)

@tf.function(experimental_compile=True)
def run_optimization(x, y):
  # Wrap computation inside a GradientTape for automatic differentiation.
  with tf.GradientTape() as g:
      pred = model(x)
      loss = cross_entropy(pred, y)
  # Compute gradients.
  gradients = g.gradient(loss, [W, b])
  # Update W and b following gradients.
  optimizer.apply_gradients(zip(gradients, [W, b]))

total_time = 0
# Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
  # Run the optimization to update W and b values.
  start = time.time_ns()
  run_optimization(batch_x, batch_y)
  end = time.time_ns()
  t = (end - start)/1000000
  
  if step != 1:
    total_time += t

  if step % display_step == 0 or step == 1:
      pred = model(batch_x)
      loss = cross_entropy(pred, batch_y)
      acc = accuracy(pred, batch_y)
      print("epoch: %i, loss: %f, accuracy: %f, time (ms): %.3f" % (step/display_step, loss, acc, t))
  # pred = model(batch_x)
  # loss = cross_entropy(pred, batch_y)
  # acc = accuracy(pred, batch_y)
  # print("epoch: %i, loss: %f, accuracy: %f, time (ms): %.3f" % (step/display_step, loss, acc, t))

# Test model
pred = model(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))
print("Total time for %i steps: %.3f ms, avg time : %.3f ms" % (training_steps-1, total_time, total_time/(training_steps-1)))
