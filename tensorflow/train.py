import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np

# Adding seed variable so random initialization is consistent
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


batch_size = 32

# Prepare input data
classes = ['dogs', 'cats']
num_classes = len(classes)

# 1/5 of the training data will be used for validation
validation_size = 0.2
img_size = 128
num_channels = 3
data_path = 'training_data'

data = dataset.read_training_dataset(data_path, img_size, classes, validation_size)

print("Finished reading input data. Printing a snippet")
print("Number of files in Training set: \t\t{}".format(len(data.train.labels)))
print("Number of files in Validation set: \t\t{}".format(len(data.valid.labels)))


sess = tf.Session()
# placeholder for training images, first parameter is None because we don't know how many images we have
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')   # one-hot encoded labels
y_true_cls = tf.argmax(y_true, axis=1)                                          # convert to actual class


## Network graph parameters
kernel_size_conv1 = 3
num_kernels_conv1 = 32

kernel_size_conv2 = 3
num_kernels_conv2 = 32

kernel_size_conv3 = 3
num_kernels_conv3 = 64

fc_layer_size = 128


def init_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def init_bias(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input, num_input_channels, kernel_size, num_kernels):
    weights = init_weights(shape=[kernel_size, kernel_size, num_input_channels, num_kernels])
    biases = init_bias(num_kernels)

    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

    layer += biases

    layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):
    # Get shape from previous layer
    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()

    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = init_weights(shape=[num_inputs, num_outputs])
    biases = init_bias(num_outputs)

    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


layer_conv1 = create_convolutional_layer(input=x,
                                         num_input_channels=num_channels,
                                         kernel_size=kernel_size_conv1,
                                         num_kernels=num_kernels_conv1)

layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                         num_input_channels=num_kernels_conv1,
                                         kernel_size=kernel_size_conv2,
                                         num_kernels=num_kernels_conv2)

layer_conv3 = create_convolutional_layer(input=layer_conv2,
                                         num_input_channels=num_kernels_conv2,
                                         kernel_size=kernel_size_conv3,
                                         num_kernels=num_kernels_conv3)

layer_flatten = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flatten,
                            num_inputs=layer_flatten.get_shape()[1:4].num_elements(),
                            num_outputs=fc_layer_size,
                            use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                            num_inputs=fc_layer_size,
                            num_outputs=num_classes,
                            use_relu=False)

y_pred = tf.nn.softmax(layer_fc2, name='y_pred')

y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.global_variables_initializer()
sess.run(init_op)


def display_progress(epoch, feed_dict_training, feed_dict_validation, val_loss):
    acc = sess.run(accuracy, feed_dict=feed_dict_training)
    val_acc = sess.run(accuracy, feed_dict=feed_dict_validation)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3: 3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

total_iterations = 0

saver = tf.train.Saver()


def train(num_iteration):
    global total_iterations

    for i in range(total_iterations, total_iterations + num_iteration):
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        feed_dict_training = {x: x_batch, y_true: y_true_batch}
        feed_dict_validation = {x: x_valid_batch, y_true: y_valid_batch}

        sess.run(optimizer, feed_dict=feed_dict_training)

        if i % int(data.train.num_examples/batch_size) == 0:
            val_loss = sess.run(cost, feed_dict=feed_dict_validation)
            epoch = int(i / int(data.train.num_examples/batch_size))

            display_progress(epoch, feed_dict_training, feed_dict_validation, val_loss)
            saver.save(sess, 'dogs-cats-model')
    total_iterations += num_iteration


train(num_iteration=1000)
sess.close()