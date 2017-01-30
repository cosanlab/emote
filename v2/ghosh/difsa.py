import os
from random import shuffle
from collections import deque
from keras import backend as K
import numpy as np

import face_pb2

class difsa_repo:

    def __init__(self, directory_path, eager=False):
        self.root_path = directory_path
        self.paths = os.listdir(self.root_path)
        shuffle(self.paths)
        self.queue = deque(self.paths)
        self.data = []

        if eager:
            self.data = self.get_dataset()

        self.numData = len(self.paths)
        self.epoch = 1

    def load_from_filepath(self, path):

        with open(path, 'rb') as fp:
            face = face_pb2.Face()
            face.ParseFromString(fp.read())
            return face

    def reset(self):
        shuffle(self.paths)
        self.queue = deque(self.paths)

    def reset_epoch(self):
        self.epoch = 1

    def get_dataset(self):
        if len(self.data) == 0:
            images = None
            labels = None
            self.reset()
            self.reset_epoch()
            epoch = self.epoch
            batch_size = 1000

            while True:
                new_images, new_labels = self.get_data(batch_size)

                if images is None:
                    images = new_images
                    labels = new_labels
                else:
                    images = np.concatenate((images, new_images))
                    labels = np.concatenate((labels, new_labels))

                if epoch != self.epoch:
                    break

            return images, labels

        else:
            self.epoch += 1
            return self.data

    def get_dataset_priors(self):
        totals = [0] * 12
        batch_size = 1000
        self.reset()
        self.reset_epoch()

        while self.get_epoch() == 1:
            images, facs = self.get_data(batch_size)

            for i in range(len(facs[0])):
                totals[i] += sum([fac[i] for fac in facs])

        return [total / float(self.numData) for total in totals]


    def get_data(self, n):

        dataPaths = []
        try:
            #Pop n items or until failure
            for i in range(n):
                dataPaths.append(self.queue.popleft())
        except IndexError, e:
            self.epoch += 1
            self.reset()

        images = []
        labels = []

        for dataPath in dataPaths:
            absDataPath = os.path.join(self.root_path, dataPath)
            data = self.load_from_filepath(absDataPath)

            image = list(bytearray(data.image))
            image_norm = [0] * len(image)
            for val, pixel in enumerate(image):
                image_norm[val] = pixel / 255.0

            image = np.asarray(image_norm)

            image = image.reshape((1, 96, 96))

            if np.count_nonzero(image.flatten()) == 0:
                print('Found empty image: %s' % dataPath)
                continue

            images.append(image)
            labels.append(np.asarray(_convert_to_one_hot(data.facs)))

        return np.asarray(images), np.asarray(labels)

    def get_epoch(self):
        return self.epoch

def _convert_to_one_hot(label):
    one_hot = [0] * len(label)
    for j, item in enumerate(label):
        if item >= 2:
            one_hot[j] = 1

    return one_hot
