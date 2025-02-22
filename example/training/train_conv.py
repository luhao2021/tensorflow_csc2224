from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_DUMP_GRAPH_PREFIX'] = '/tmp/tf_graphs/'
os.environ['XLA_FLAGS'] = "--xla_dump_hlo_as_text --xla_dump_to=/tmp/tf_graphs/xla_dump"

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# dtype = tf.float32
dtype = tf.cus


# MNIST dataset parameters.
num_classes = 10 # total classes (0-9 digits).

# Training parameters.
learning_rate = 0.001
training_steps = 200
batch_size = 128
display_step = 10

# Network parameters.
conv1_filters = 32 # number of filters for 1st conv layer.
conv2_filters = 64 # number of filters for 2nd conv layer.
fc1_units = 1024 # number of neurons for 1st fully-connected layer.

# Prepare MNIST data.
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.

# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

# Create some wrappers for simplicity.
@tf.function(experimental_compile=True)
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation.
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

@tf.function(experimental_compile=True)
def maxpool2d(x, k=2):
    # MaxPool2D wrapper.
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# Store layers weight & bias

# A random value generator to initialize weights.
random_normal = tf.initializers.RandomNormal()

weights = {
    # Conv Layer 1: 5x5 conv, 1 input, 32 filters (MNIST has 1 color channel only).
    'wc1': tf.Variable(random_normal([5, 5, 1, conv1_filters])),
    # Conv Layer 2: 5x5 conv, 32 inputs, 64 filters.
    'wc2': tf.Variable(random_normal([5, 5, conv1_filters, conv2_filters])),
    # FC Layer 1: 7*7*64 inputs, 1024 units.
    'wd1': tf.Variable(random_normal([7*7*64, fc1_units])),
    # FC Out Layer: 1024 inputs, 10 units (total number of classes)
    'out': tf.Variable(random_normal([fc1_units, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.zeros([conv1_filters])),
    'bc2': tf.Variable(tf.zeros([conv2_filters])),
    'bd1': tf.Variable(tf.zeros([fc1_units])),
    'out': tf.Variable(tf.zeros([num_classes]))
}

# Create model
@tf.function(experimental_compile=True)
def conv_net(x):
    # Input shape: [-1, 28, 28, 1]. A batch of 28x28x1 (grayscale) images.
    x = tf.cast(tf.reshape(x, [-1, 28, 28, 1]), dtype)
    wc1 = tf.cast(weights['wc1'], dtype)
    wc2 = tf.cast(weights['wc2'], dtype)
    wd1 = tf.cast(weights['wd1'], dtype)
    wout = tf.cast(weights['out'], dtype)

    bc1 = tf.cast(biases['bc1'], dtype)
    bc2 = tf.cast(biases['bc2'], dtype)
    bd1 = tf.cast(biases['bd1'], dtype)
    bout = tf.cast(biases['out'], dtype)

    # Convolution Layer. Output shape: [-1, 28, 28, 32].
    conv1 = conv2d(x, wc1, bc1)
    
    # Max Pooling (down-sampling). Output shape: [-1, 14, 14, 32].
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer. Output shape: [-1, 14, 14, 64].
    conv2 = conv2d(conv1, wc2, bc2)
    
    # Max Pooling (down-sampling). Output shape: [-1, 7, 7, 64].
    conv2 = maxpool2d(conv2, k=2)

    # Reshape conv2 output to fit fully connected layer input, Output shape: [-1, 7*7*64].
    fc1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]])
    
    # Fully connected layer, Output shape: [-1, 1024].
    fc1 = tf.add(tf.matmul(fc1, wd1), bd1)
    # Apply ReLU to fc1 output for non-linearity.
    fc1 = tf.maximum(fc1, 0)
    # fc1 = tf.nn.relu(fc1)

    # Fully connected layer, Output shape: [-1, 10].
    out = tf.add(tf.matmul(fc1, wout), bout)
    out = tf.cast(out, tf.float32)
    # Apply softmax to normalize the logits to a probability distribution.
    return tf.nn.softmax(out)

# Cross-Entropy loss function.
def cross_entropy(y_pred, y_true):
    # Encode label to a one hot vector.
    y_true = tf.one_hot(y_true, depth=num_classes)
    # Clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # Compute cross-entropy.
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

# ADAM optimizer.
optimizer = tf.optimizers.Adam(learning_rate)

# Optimization process. 
@tf.function(experimental_compile=True)
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = conv_net(x)
        loss = cross_entropy(pred, y)
        
    # Variables to update, i.e. trainable variables.
    trainable_variables = list(weights.values()) + list(biases.values())

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)
    
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))

# Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y)
    
    if step % display_step == 0:
        pred = conv_net(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))