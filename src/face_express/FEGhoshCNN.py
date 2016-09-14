import os
import logging
import tensorflow as tf

from util import paths
from FEExpresser import FEExpresser
from util.tf_util import weight_variable, bias_variable, conv2d, max_pool_2x2

KERNEL_SIZE=5
LEARNING_RATE=0.01
WEIGHT_STD = 0.3
MODEL_NAME = 'GhoshCNN'
BATCH_SIZE = 64

FULL_SIZE_1 = 500
FULL_SIZE_2 = 100
FULL_SIZE_3 = 12


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

        #If the model has already been trained, load it again
        if False and os.path.isfile(paths.get_saved_model_path(MODEL_NAME)):
            logging.info("Found saved model, restoring variables....")
            self.saver.restore(self.session, paths.get_saved_model_path(MODEL_NAME))
            self.isTrained = True

    def train(self, save=True):
        logging.info("Beginning training session for MultiLabelCNN")

        images, labels = self.repo.get_training_batch(BATCH_SIZE)
        i = 0

        #Do training
        correct_prediction = tf.equal(self.output, self.y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)
            
        while len(images) > 0:

            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={self.x: images,
                         self.y_: labels,
                         self.keep_prob: 1.0})

                logging.info("Accuracy at %d: %g" % (i, train_accuracy))

             #Do training
            feed_dict = {self.x: images,
                         self.y_: labels,
                         self.keep_prob: 0.5}
            self.session.run(self.trainer,
                             feed_dict=feed_dict)

            images, labels = self.repo.get_training_batch(BATCH_SIZE)

            i += 1


        if save:
            self.saver.save(self.session, paths.get_saved_model_path(MODEL_NAME))

        #train_classifier(self)

    def train_classifier(self):
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
        conv_W_2 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, 70, 10], name='conv_W_2')
        conv_b_2 = bias_variable([10], name="conv_b_2")
        conv_2 = conv2d(max_pool_1, conv_W_2, 'conv_layer_2')
        relu_2 = tf.nn.relu(conv_2 + conv_b_2, 'relu_2')
        drop_2 = tf.nn.dropout(relu_2, self.keep_prob, name='drop_2')
        max_pool_2 = max_pool_2x2(drop_2, name='pool_2')

        #Fully connected layers
        
        flattened = tf.reshape(max_pool_2, [-1, 24*24*10], name='layer3_pool')

        #1st fully connected
        flat_W_1 = weight_variable(shape=[24*24*10, FULL_SIZE_1], name='flat_W_1')
        flat_b_1 = bias_variable(shape=[FULL_SIZE_1], name='flat_b_1')
        flat_1 = tf.nn.relu(tf.matmul(flattened, flat_W_1) + flat_b_1, name='flat_1')

        #2nd
        flat_W_2 = weight_variable(shape=[FULL_SIZE_1, FULL_SIZE_2], name='flat_W_2')
        flat_b_2 = bias_variable(shape=[FULL_SIZE_2], name='flat_b_2')
        flat_2 = tf.nn.relu(tf.matmul(flat_1, flat_W_2) + flat_b_2, name='flat_2')

        #3rd
        flat_W_3 = weight_variable(shape=[FULL_SIZE_2, FULL_SIZE_3], name='flat_W_3')
        flat_b_3 = bias_variable(shape=[FULL_SIZE_3], name='flat_b_3')
        flat_3 = tf.nn.relu(tf.matmul(flat_2, flat_W_3) + flat_b_3, name='flat_3')

        cost = self._multilabel_error(self.y_, flat_3)

        tf.scalar_summary('cost/summary', cost)
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

        #initialize the variables of the graph
        init_ops = tf.initialize_all_variables()

        return train_step, flat_3, init_ops

    def _multilabel_error(self, label, output):
        p_hat = tf.exp(output) / tf.reduce_sum(tf.exp(output))
        return -tf.reduce_mean(tf.reduce_sum(label * tf.log(p_hat)))


    def get_image_size(self):
        raise 96
