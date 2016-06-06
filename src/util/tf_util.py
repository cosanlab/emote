import tensorflow as tf
from random import randint

"""
Helper functions for instantiating different tensors or other common lines in tensorflow
"""

def weight_variable(shape, stddev=0.3, name='weight_var'):
    '''Produces variable from a random_normal

    :param shape: Shape of desired var
    :type shape: Tensor
    :param stddev: desired standard deviation of normal
    :type stddev: float
    :param name: Name of tensor used in summaries
    :type name: str
    :returns: Variable
    '''
    initial = tf.random_normal(shape, stddev=stddev)
    var = tf.Variable(initial, name=name)
    variable_summaries(var, name + '/summary')
    return var

def bias_variable(shape, name='bias_var'):
    '''Produces variable with initial value of 0.1

    :param shape: Shape of desired var
    :type shape: Tensor
    :param name: Name of tensor used in summaries
    :type name: str
    :returns: Variable
    '''
    initial = tf.constant(0.1, shape=shape)
    var = tf.Variable(initial, name=name)
    variable_summaries(var, name + '/summary')
    return var

def conv2d(x, W, name='conv_var'):
    """A 2D convolution operation on an input x and kernel W, with stride size or 1

    :param x: Convolution input
    :type x: tensor
    :param W: Convolution kernel
    :type W: tensor
    :param name: Name of tensor used in summaries
    :type name: str
    :returns: Tensor - output of convolution
    """
    conv = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name=name)
    variable_summaries(conv, name + '/summary')
    return conv

def max_pool_2x2(x, name='max_pool'):
    """ Performs a max_pool operation on input

    :param x: Input tensor
    :type x: tensor
    :param name: Name of tensor used in summaries
    :type name: str
    :returns: Tensor - pooled output
    """
    pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    variable_summaries(pool, name + '/summary')
    return pool

def mean_square_error(predictions, labels):
    """ Calculates mean-square error over two tensors

    :param predictions: Prediction tensor
    :type predictions: Tensor
    :param labels: Labels tensor
    :type labels: Tensor
    :returns: Tensor - Mean-square error of inputs
    """
    return tf.reduce_mean(tf.square(predictions - labels))

def variable_summaries(var, name):
    """Sets up various summaries on a Variable

    :param var: Variable to be summarized
    :type var: Variable
    :param name: Name of sumamry
    :type name: str
    """
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

