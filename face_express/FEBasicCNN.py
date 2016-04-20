import os
import logging
import tensorflow as tf

from FEExpresser import FEExpresser
from util.tf_util import weight_variable, bias_variable, conv2d
from util.paths import get_project_home


 # Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999   # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
BATCH_SIZE = 50
MODEL_NAME = 'FEBasicCNN.ckpt'

class FEBasicCNN(FEExpresser):

    def __init__(self, repo, codes, intensity=False):
        logging.info("Initializing FEBasicCNN")
        self.repo = repo
        self.codes = codes
        self.intensity = intensity
        self.isTrained = False

        self.x = tf.placeholder(tf.float32, shape=(None, self.repo.get_height(), self.repo.get_width(), 3), name='x_var')
        self.y_ = tf.placeholder(tf.float32, shape=(None, len(self.codes)), name='y_var')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')


        logging.info("Creating network variables")
        self.trainer, self.output, self.ops = self._create_network()

        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(self.ops)

        if os.path.isfile(self.get_model_path()):
            logging.info("Found saved model, restoring variables....")
            self.saver.restore(self.session, self.get_model_path())
            self.isTrained = True

    def predict(self, face):
        if self.isTrained:
            prediction = self.session.run(self.output, feed_dict={self.x:face})
            logging.info("Predicted: " + str(prediction))
            return prediction

        return None

    def train(self):
        logging.info("Beginning training session")

        items = self.repo.get_items(BATCH_SIZE)
        i = 0

        correct_prediction = tf.equal(tf.argmax(self.output,1), tf.argmax(self.y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while len(items) > 0:
            images = [fac.get_image() for fac in items]
            label_sets = [fac.get_labels() for fac in items]
            labels = []

            #Convert FAC lists into equal size code lists
            for set in label_sets:
                new_label = []
                for i, val in enumerate(self.codes):
                    if val in set:
                        new_label.append(1)
                    else:
                        new_label.append(0)

                labels.append(new_label)


            if i % 2 == 0:
                train_accuracy = accuracy.eval(session=self.session, feed_dict={self.x:images, self.y_:labels , self.keep_prob: 1.0})
                logging.info("Training: step %d, training accuracy %g"%(i, train_accuracy))

            self.trainer.run(session=self.session, feed_dict={self.x:images, self.y_:labels, self.keep_prob: 0.5})
            i += 1
            items = self.repo.get_items(BATCH_SIZE)

        self.isTrained = True
        logging.info("Training complete")
        self.saver.save(self.session, self.get_model_path())

    def _create_network(self):

        layer1_conv_W = weight_variable([5,5,3,32], name="layer1_W")
        layer1_conv_b = bias_variable([32], name="layer1_b")

        layer1_conv = tf.nn.relu(conv2d(self.x, layer1_conv_W, name='layer1_conv') + layer1_conv_b, name='layer1_relu')

        layer2_conv_W = weight_variable([5,5,32,64], name='layer2_W')
        layer2_conv_b = bias_variable([64], name='layer2_b')

        layer2_conv = tf.nn.relu(conv2d(layer1_conv, layer2_conv_W, ) + layer2_conv_b, name='layer2_relu')

        layer3_pool = tf.nn.max_pool(layer2_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='layer3_pool')

        layer4_conv_W = weight_variable(shape = [4,4,64,128], name='layer4_W')
        layer4_conv_b = bias_variable([128], name='layer4_b')

        layer4_conv = tf.nn.relu(conv2d(layer3_pool, layer4_conv_W, name='layer4_conv') + layer4_conv_b, name='layer4_relu')

        layer5_pool = tf.nn.max_pool(layer4_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='layer5_pool')

        layer6_full_W = weight_variable(shape= [ (self.repo.get_height() / 4) * (self.repo.get_width() / 4) * 128, 1024], name='layer6_W')
        layer6_full_b = bias_variable([1024], name='layer6_b')

        layer5_pool_flat = tf.reshape(layer5_pool, [-1, (self.repo.get_height() / 4) * (self.repo.get_width() / 4) * 128], name='layer5_pool_flat')
        layer6_full = tf.nn.relu(tf.matmul(layer5_pool_flat, layer6_full_W, name='layer6_matmull') + layer6_full_b, name='layer6_full')

        layer6_full_drop = tf.nn.dropout(layer6_full, self.keep_prob, name='layer6_drop')

        layer7_soft_W = weight_variable([1024, len(self.codes)], name='layer7_W')
        layer7_soft_b = bias_variable([len(self.codes)], name='layer7_b')
        layer7_soft = tf.nn.softmax(tf.matmul(layer6_full_drop, layer7_soft_W, name='layer7_matmull') + layer7_soft_b, name='layer7_soft')

        cross_entropy = -tf.reduce_sum(self.y_*tf.log(layer7_soft))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        init_ops = tf.initialize_all_variables()

        return train_step, layer7_soft, init_ops

    def get_model_path(self):
        return get_project_home() + '/data/saved_models/' + MODEL_NAME





