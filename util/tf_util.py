import tensorflow as tf
from random import randint

def weight_variable(shape, stddev=0.1, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W, name=None):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name=name)

def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def mean_square_error(predictions, labels):
    return tf.reduce_mean(tf.square(predictions - labels))

def std_dev(inputs):
    return (randint(2, 12) / 10.0) / inputs