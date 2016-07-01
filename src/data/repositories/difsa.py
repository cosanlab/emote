import cv2
import csv
import os
import random
from data.repositories import FACRepository
from util.paths import DataLoc
import util.constants as ks
from util import config
import tensorflow as tf

BUFFER_SIZE = 50
AUS = [1,2,4,5,6,9,12,15,17,20,25,26]
IMAGE_SIZE = 96
TRAINING_DIR = 'training'
TESTING_DIR = 'testing'
VALIDATION_DIR = 'validation'

class DIFSARepository(FACRepository):
    """Repository to manage loading the DIFSA dataset

    Internally organized by AU in order to return data points that only contain specific AUs
    """

    def __init__(self):
        path_finder = DataLoc()

        self.data_dir = path_finder.get_path(ks.kDataDIFSA)
        self.training_path = os.path.join(self.data_dir, TRAINING_DIR)
        self.testing_path = os.path.join(self.data_dir, TESTING_DIR)
        self.validation_path = os.path.join(self.data_dir, VALIDATION_DIR)

        training_files = os.listdir(self.training_path)
        testing_files = os.listdir(self.testing_path)
        validation_files = os.listdir(self.validation_path)

        self.testing_size = len(testing_files)
        self.validation_size = len(validation_files)

        self.training_queue = tf.train.string_input_producer(training_files, shuffle=True)
        self.testing_queue = tf.train.string_input_producer(testing_files, shuffle=True)
        self.validation_queue = tf.train.string_input_producer(validation_files, shuffle=True)

        FACRepository.__init__(self, BUFFER_SIZE)


    def _read_and_decode(self, queue, batch_size):

        reader = tf.TFRecordReader()
        _, serialized_examples = reader.read_up_to(self.training_queue, batch_size)

        features = []
        labels = []

        for i in range(batch_size):
            feature = tf.parse_example(serialized_examples[i], features={
                'label': tf.FixedLenFeature([], tf.int64),
                'image': tf.FixedLenFeature([], tf.string)
            })

            image = tf.decode_raw(features['image'], tf.uint8)
            image.set_shape([IMAGE_SIZE])

            features.append(feature['image'])
            labels.append(feature['label'])

        return features, labels


    def get_training_batch(self, batch_size=None, ):
        """Get next batch of training examples

        :param batch_size: (int) size of the batch to be returned
        :return: list of training examples and features
        """
        if batch_size is None:
            batch_size = self.batch_size

        return self._read_and_decode(self.training_queue, batch_size)

    def get_validation_data(self):
        """List of validation data

        :return: Validation data
        """

        return self._read_and_decode(self.validation_queue, self.validation_size)


    def get_testing_data(self):
        """List of testing data
        :returns: List of testing data
        """

        return self._read_and_decode(self.testing_queue, self.testing_size)


def main():
    ck = DIFSARepository()

    items = ck.get_items(20)

    for i in xrange(5):
        items = ck.get_items(50)

    uni = set(items)

    print("Buffer Size: " + str(len(ck.data_buffer)))
    print("Item length: " + str(len(items)))
    print("Set size: " + str(len(uni)))

if __name__ == '__main__':
    main()
