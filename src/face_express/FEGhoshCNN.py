import os
import logging
import tensorflow as tf

from FEExpresser import FEExpresser
from util.tf_util import weight_variable, bias_variable, conv2d, max_pool_2x2

KERNEL_SIZE=5
LEARNING_RATE=0.01
WEIGHT_STD = 0.3

FULL_SIZE_1 = 500
FULL_SIZE_2 = 100
FULL_SIZE_3 = 10


class FEGhoshCNN(FEExpresser):
    ''' Implements a the architecture described in
    http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=7344632&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel7%2F7332343%2F7344542%2F07344632.pdf%3Farnumber%3D7344632
    '''

    def __init__(self, image_size, codes, repo=None):
        """ Intializes

        :param image_size: The rectangular length the model should expect
        :type image_size: int
        :param codes: List of action units to train on and recognize
        :type codes: [int]
        :param repo: The data repository to pull training examples from. Not necessary to provide on pre-trained models
        :type repo: data.repositories.FACRepository
        """

        logging.info("Initializing FEMultiLabelCNN")
        self.repo = repo
        self.codes = codes
        self.image_size = image_size
        self.isTrained = False

        #Model placholders
        self.x = tf.placeholder(tf.float32, shape=(None, self.image_size, self.image_size, 1), name='x_var')
        self.y_ = tf.placeholder(tf.float32, shape=(None, len(self.codes)), name='y_var')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        #Graph variables
        logging.info("Creating network variables")
        self.trainer, self.output, self.ops = self._create_network()

        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(self.ops)

    def train(self):
        '''Trains the expression recognition model
            Must be overriden by any extendng class

            :raises: NotImplementedError
        '''
        raise NotImplementedError

    def predict(self, face):
        ''' Given a face image, predicts the corresponding expression

        :param face: Image which matches in size with the parameters of the model
        :type face: OpenCV Mat (numpy ndarray would also probably work)
        :returns: list of expression data
        '''
        raise NotImplementedError

    def _create_network(self):
        logging.info("Begin creating Ghosh graph")

        #First "mega" layer 
        conv_W_1 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, 1, 70], name='conv_W_1')
        conv_b_1 = bias_variable([70], name="conv_b_1")
        conv_1 = conv2d(self.x, conv_W_1, 'conv_layer_1')
        relu_1 = tf.nn.relu(conv_1 + conv_b_1, 'relu_1')
        drop_1 = tf.nn.dropout(relu_1, self.keep_prob, name='drop_1')
        max_pool_1 = max_pool_2x2(drop_1, name='pool_1')

        #Second "mega" layer 
        conv_W_2 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, 1, 10], name='conv_W_2')
        conv_b_2 = bias_variable([70], name="conv_b_2")
        conv_2 = conv2d(max_pool_1, conv_W_2, 'conv_layer_2')
        relu_2 = tf.nn.relu(conv_2 + conv_b_2, 'relu_2')
        drop_2 = tf.nn.dropout(relu_2, self.keep_prob, name='drop_2')
        max_pool_2 = max_pool_2x2(drop_2, name='pool_2')

        #Fully connected layers
        
        flattened = tf.reshape(max_pool_2, [-1], name='layer3_pool')

        #1st fully connected
        flat_W_1 = weight_variable(shape=[FULL_SIZE_1, size(flattened)], name='flat_W_1')
        flat_b_1 = bias_variable(shape=[FULL_SIZE_1], name='flat_b_1')
        flat_1 = tf.nn.relu(tf.matmul(flat_W_1, flattened) + flat_b_1, name='flat_1')

        #2nd
        flat_W_2 = weight_variable(shape=[FULL_SIZE_2, FULL_SIZE_1], name='flat_W_2')
        flat_b_2 = bias_variable(shape=[FULL_SIZE_2], name='flat_b_2')
        flat_2 = tf.nn.relu(tf.matmul(flat_W_2, flat_1) + flat_b_2, name='flat_2')

        #3rd
        flat_W_3 = weight_variable(shape=[FULL_SIZE_3, FULL_SIZE_2], name='flat_W_3')
        flat_b_3 = bias_variable(shape=[FULL_SIZE_3], name='flat_b_3')
        flat_3 = tf.nn.relu(tf.matmul(flat_W_3, flat_2) + flat_b_3, name='flat_3')

        cost = self._multilabel_error(self.y_, flat_3)

        tf.scalar_summary('cost/summary', cost)
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

        #initialize the variables of the graph
        init_ops = tf.initialize_all_variables()

        return train_step, flat_3, init_ops

    def _multilabel_error(self, label, output):
        p_hat = tf.exp(output) / tf.sum(tf.exp(output))
        return -tf.reduce_mean(tf.sum(label) * tf.log(p_hat))


    def get_image_size(self):
        raise 96