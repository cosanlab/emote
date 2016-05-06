import tensorflow as tf
from random import randint

def weight_variable(shape, stddev=0.3, name='weight_var'):
    initial = tf.random_normal(shape, stddev=stddev)
    var = tf.Variable(initial, name=name)
    variable_summaries(var, name + '/summary')
    return var

def bias_variable(shape, name='bias_var'):
    initial = tf.constant(0.1, shape=shape)
    var = tf.Variable(initial, name=name)
    variable_summaries(var, name + '/summary')
    return var

def conv2d(x, W, name='conv_var'):
    conv = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name=name)
    variable_summaries(conv, name + '/summary')
    return conv

def max_pool_2x2(x, name='max_pool'):
    pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    variable_summaries(pool, name + '/summary')
    return pool

def mean_square_error(predictions, labels):
    return tf.reduce_mean(tf.square(predictions - labels))

def variable_summaries(var, name):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

