import os
import logging
import tensorflow as tf

from FEExpresser import FEExpresser
from util.tf_util import weight_variable, bias_variable, conv2d, max_pool_2x2



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
        :raises: NotImplementedError
        '''
        raise NotImplementedError

    def _create_network(self):

        print("Begin creating graph")


    def _multilabel_error(self, label, output):
        p_hat = tf.exp(output) / tf.sum(tf.exp(output))
        return -tf.reduce_mean(tf.sum(label) * tf.log(p_hat))



    def get_image_size(self):

        raise 96