import os
import logging
import tensorflow as tf

from FEExpresser import FEExpresser
from util.tf_util import weight_variable, bias_variable, conv2d, std_dev
from util import paths


 # Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999   # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 1e-5  # Learning rate decay factor.
LEARNING_RATE = 3.0       # Initial learning rate.
MOMENTUM = 0.9
BATCH_SIZE = 64
MODEL_NAME = 'FESingleAUCNN.ckpt'

class FEMultiLabelCNN(FEExpresser):

    def __init__(self, image_size, codes, repo=None):
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
        if self.isTrained:
            face4d = face.reshape([1, self.image_size, self.image_size, 1])
            prediction = self.session.run(self.output, feed_dict={self.x:face4d,
                                                                  self.keep_prob:1.0})
            logging.info("Predicted: " + str(prediction))
            return prediction

        return None

    def get_image_size(self):
        return self.image_size

    def train(self):
        logging.info("Beginning training session for MultiLabelCNN")

        items = self.repo.get_items(BATCH_SIZE, nonzero=True)
        i = 0

        correct_prediction = tf.equal(self.output, self.y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while len(items) > 0:

            images, labels = self._items_to_lists(items)

            if i % 5 == 0:
                #Check Accuracy
                train_accuracy = accuracy.eval(session=self.session,
                                               feed_dict={self.x: images,
                                                          self.y_: labels ,
                                                          self.keep_prob: 1.0})
                logging.info("Training: step %d, accuracy: %g"%(i, train_accuracy))

                prediction = self.output.eval(session=self.session,
                                 feed_dict={self.x: [images[0]],
                                            self.y_:[labels[0]],
                                            self.keep_prob: 1.0})

                print("Label: " + str(labels[0]))
                print("Prediction: " + str(prediction[0]))

                positive = 0
                for lab in labels:
                    positive = positive + 1 if lab[1] == 1 else positive

                print("Percent positive: " + str(positive/float(len(labels))))


            #Do training
            self.session.run(self.trainer,
                             feed_dict={self.x: images,
                                        self.y_: labels,
                                        self.keep_prob: 0.5})
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
        #self.saver.save(self.session, paths.get_saved_model_path(MODEL_NAME))


    def _create_network(self):

        layer1_conv_W = weight_variable([5,5,1,64], stddev=std_dev(self.image_size*self.image_size), name="layer1_W")
        layer1_conv_b = bias_variable([64], name="layer1_b")
        layer1_conv   = tf.nn.relu(conv2d(self.x, layer1_conv_W, name='layer1_conv') + layer1_conv_b, name='layer1_relu')
        layer1_pool   = tf.nn.max_pool(layer1_conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='layer1_pool')
        # layer1_drop   = tf.nn.dropout(layer1_pool, self.keep_prob, name='layer1_drop')

        layer2_conv_W = weight_variable([5,5,64,128], stddev=std_dev(64), name='layer2_W')
        layer2_conv_b = bias_variable([128], name='layer2_b')
        layer2_conv   = tf.nn.relu(conv2d(layer1_pool, layer2_conv_W) + layer2_conv_b, name='layer2_relu')
        layer2_pool   = tf.nn.max_pool(layer2_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='layer2_pool')

        image_size_8  = self.image_size / 8
        layer3_conv_W = weight_variable([5,5,128,256], stddev=std_dev(128), name='layer2_W')
        layer3_conv_b = bias_variable([256], name='layer2_b')
        layer3_conv   = tf.nn.relu(conv2d(layer2_pool, layer3_conv_W) + layer3_conv_b, name='layer2_relu')
        layer3_pool   = tf.nn.max_pool(layer3_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='layer2_pool')
        # layer3_drop   = tf.nn.dropout(layer3_pool, self.keep_prob, name='layer3_drop')

        layer3_flat = tf.reshape(layer3_pool, [-1, image_size_8 * image_size_8 * 256], name='layer2_pool_flat')

        layer4_full_W = weight_variable(shape=[image_size_8 * image_size_8 * 256, 300], name='layer3_W')
        layer4_full_b = bias_variable([300], name='layer3_b')
        layer4_full   = tf.nn.relu(tf.matmul(layer3_flat, layer4_full_W, name='layer6_matmull') + layer4_full_b, name='layer4_full')
        layer4_drop   = tf.nn.dropout(layer4_full, self.keep_prob, name='layer4_drop')

        layer5_sig_W = weight_variable(shape=[300, len(self.codes)], name='layer3_W')
        layer5_sig_b = bias_variable([len(self.codes)], name='layer3_b')
        layer5_sig   = tf.nn.sigmoid(tf.matmul(layer4_drop, layer5_sig_W) + layer5_sig_b)

        cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(layer5_sig, self.y_))
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
        init_ops = tf.initialize_all_variables()

        return train_step, layer5_sig, init_ops

    def _items_to_lists(self, items):
        images = []
        labels = []

        for fac in items:
            images.append(fac.get_image())
            new_labels = []
            for code in self.codes:
                if code in fac.get_labels():
                    new_labels.append(1)
                else:
                    new_labels.append(0)

            labels.append(new_labels)

        return images, labels

def main():
    print("Test")
    e
if __name__ == '__main__':
    main()