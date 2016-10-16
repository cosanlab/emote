import os
from random import shuffle
from collections import deque
from keras import backend as K
import numpy as np

import face_pb2

class difsa_repo:

    def __init__(self, directory_path):
        self.root_path = directory_path
        self.paths = os.listdir(self.root_path)
        shuffle(self.paths)
        self.queue = deque(self.paths)

        self.numData = len(self.paths)
        self.dataPresented = 0

    def load_from_filepath(self, path):

        with open(path, 'rb') as fp: 
            face = face_pb2.Face()
            face.ParseFromString(fp.read())
            return face

    def get_data(self, n):
        self.dataPresented += n

        dataPaths = [self.queue.popleft() for i in range(n)]
        self.queue.extend(dataPaths)

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
            images.append(image)
            labels.append(np.asarray(_convert_to_one_hot(data.facs)))

            self.dataPresented += 1
            if self.dataPresented % self.numData == 0:
                shuffle(self.paths)
                self.queue = deque(self.paths)

        return images, labels

    def get_epoch(self):
        return (self.dataPresented / self.numData) + 1

def _convert_to_one_hot(label):
    one_hot = [0] * len(label)
    for j, item in enumerate(label):
        if item >= 2:
            one_hot[j] = 1

    return one_hot


