import os
import logging
import tensorflow as tf

from FEExpresser import FEExpresser
from util.tf_util import weight_variable, bias_variable, conv2d, max_pool_2x2


 # Constants describing the training process.
LEARNING_RATE = 1e-5     # Initial learning rate.
MOMENTUM = 0.9
BATCH_SIZE = 32
MODEL_NAME = 'FESingleAUCNN.ckpt'

class FEMultiLabelCNN(FEExpresser):
    """ A convolution neural network setup to recognize mulitple AU labels in one image
    """

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
        if os.path.isfile(paths.get_saved_model_path(MODEL_NAME)):
            logging.info("Found saved model, restoring variables....")
            self.saver.restore(self.session, paths.get_saved_model_path(MODEL_NAME))
            self.isTrained = True

    def predict(self, face):
        """ Given a face image, predicts the AUs present in the image

        :param face: Image with a face
        :type face: OpenCV Mat or numpy ndarray

        :returns: [float] - predictions
        """
        if self.isTrained:
            face4d = face.reshape([1, self.image_size, self.image_size, 1])
            prediction = self.session.run(self.output, feed_dict={self.x:face4d, self.keep_prob:1.0})logging.info("Predicted: " + str(prediction))
            return prediction

        return None

    def get_image_size(self):
        """ Returns the size of the rectangular image the model is setup to handle

        :returns: int - if returns n, the image has dimensions n x n
        """
        return self.image_size

    def train(self):
        """Trains the CNN with items from the repo provided with __init__
        Logs accuracy progress every 5 epochs to logfile
        """
        logging.info("Beginning training session for MultiLabelCNN")

        items = self.repo.get_items(BATCH_SIZE, nonzero=True)
        i = 0

        #Determine how a prediction is correct, and how accurate it is
        correct_prediction = tf.equal(self.output, self.y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)

        #Setup summaries for tensorboard
        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(paths.get_project_home() + '/logs/summaries', self.session.graph)

        #Continue training as long as the repository is return items
        while len(items) > 0:

            #Get new items
            images, labels = self._items_to_lists(items)

            if i % 5 == 0:
                #Check accuracy on never-before-seen examples
                train_accuracy = accuracy.eval(session=self.session,
                                               feed_dict={self.x: images,
                                                          self.y_: labels ,
                                                          self.keep_prob: 1.0})
                logging.info("Training: step %d, accuracy: %g"%(i, train_accuracy))

                prediction = self.output.eval(session=self.session,
                                 feed_dict={self.x: [images[0]],
                                            self.y_:[labels[0]],
                                            self.keep_prob: 1.0})

                logging.info("Label: " + str(labels[0]))
                logging.info("Prediction: " + str(prediction[0]))
                logging.info("----------------------------------")


            #Do training
            feed_dict = {self.x: images,
                         self.y_: labels,
                         self.keep_prob: 0.5}
            self.session.run(self.trainer,
                             feed_dict=feed_dict)

            #Record summary for this epoch
            summary = self.session.run(merged, feed_dict=feed_dict)
            train_writer.add_summary(summary, 1)

            #Update for next epoch
            i += 1
            items = self.repo.get_items(BATCH_SIZE, nonzero=True)

        #Test
        test_items = self.repo.get_testing_items()
        images, labels = self._items_to_lists(test_items)


        train_accuracy = accuracy.eval(session=self.session,
                                       feed_dict={self.x: images,
                                                  self.y_: labels ,
                                                  self.keep_prob: 1.0})
        logging.info("Testing Accuracy: %g" % train_accuracy)

        self.isTrained = True
        logging.info("Training complete")
        #Uncomment below to save model for later!
        #


    def _create_network(self):
        """Sets up the graph for the neural network

        :returns: (tensor) train_step - tensor to train on, (tensor) output - raw output of network, (ops) init_ops - set of graph operations
        """

        #Layer 1 convolution + 2x2 maxpool
        layer1_conv_W = weight_variable([5,5,1,64], name="layer1_W")
        layer1_conv_b = bias_variable([64], name="layer1_b")
        layer1_conv   = tf.nn.relu(conv2d(self.x, layer1_conv_W, name='layer1_conv') + layer1_conv_b, name='layer1_relu')
        tf.histogram_summary('layer1_relu', layer1_conv)
        layer1_pool   = max_pool_2x2(layer1_conv, name='layer1_pool')
        # layer1_drop   = tf.nn.dropout(layer1_pool, self.keep_prob, name='layer1_drop')

        #Layer 2 convolution + 2x2 maxpool
        layer2_conv_W = weight_variable([5,5,64,128], name='layer2_W')
        layer2_conv_b = bias_variable([128], name='layer2_b')
        layer2_conv   = tf.nn.relu(conv2d(layer1_pool, layer2_conv_W, name='layer2_conv') + layer2_conv_b, name='layer2_relu')
        tf.histogram_summary('layer2_relu', layer2_conv)
        layer2_pool   = max_pool_2x2(layer2_conv, name='layer2_pool')

        #Layer 3 convolution + 2x2 maxpool
        image_size_8  = self.image_size / 8
        layer3_conv_W = weight_variable([5,5,128,256], name='layer3_W')
        layer3_conv_b = bias_variable([256], name='layer3_b')
        layer3_conv   = tf.nn.relu(conv2d(layer2_pool, layer3_conv_W, name='layer3_conv') + layer3_conv_b, name='layer3_relu')
        tf.histogram_summary('layer3_relu', layer3_conv)
        layer3_pool   = max_pool_2x2(layer3_conv, name='layer3_pool')

        # layer3_drop   = tf.nn.dropout(layer3_pool, self.keep_prob, name='layer3_drop')

        #Flatten output to 1D tensor
        layer3_flat = tf.reshape(layer3_pool, [-1, image_size_8 * image_size_8 * 256], name='layer3_pool')

        #Fully connected layer (image_size_8^2 * 256 -> 300 neurons)
        layer4_full_W = weight_variable(shape=[image_size_8 * image_size_8 * 256, 300], name='layer4_W')
        layer4_full_b = bias_variable([300], name='layer4_b')
        layer4_full   = tf.nn.relu(tf.matmul(layer3_flat, layer4_full_W, name='layer4_matmull') + layer4_full_b, name='layer4_full')
        tf.histogram_summary('layer4_relu', layer4_full)
        layer4_drop   = tf.nn.dropout(layer4_full, self.keep_prob, name='layer4_drop')

        #Fully connected layer (300 -> len(codes) neurons)
        layer5_relu_W = weight_variable(shape=[300, len(self.codes)], name='layer5_W')
        layer5_relu_b = bias_variable([len(self.codes)], name='layer5_b')
        layer5_relu   = tf.nn.relu(tf.matmul(layer4_drop, layer5_relu_W) + layer5_relu_b, name='layer5_relu')
        tf.histogram_summary('layer5_relu', layer5_relu)

        #Cost function and reduction
        sigmoids = tf.nn.sigmoid_cross_entropy_with_logits(layer5_relu, self.y_)
        cost = tf.reduce_mean(sigmoids)

        #Setup training
        tf.scalar_summary('cost/summary', cost)
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

        #initialize the variables of the graph
        init_ops = tf.initialize_all_variables()

        return train_step, layer5_relu, init_ops

    def _items_to_lists(self, items):
        """Converts FAC items to standard list data to be used as labels

        ToDo: This should probably be a method of the repository

        :param items: A list of FAC items provided from a repository
        :type items: [data.data_types.FAC]
        :returns: ([ndarray]) images, ([float]) labels
        """
        images = []
        labels = []

        for fac in items:
            new_labels = []
            for code in self.codes:
                if fac.has_au(code):
                    new_labels.append(1)
                else:
                    new_labels.append(0)

            if len(list(set(new_labels))) > 1:
                images.append(fac.get_image())
                labels.append(new_labels)

        return images, labels


def main():
    print("Test")
    
if __name__ == '__main__':
    main()