import os
from random import shuffle
from collections import deque

import face_pb2

class difsa_repo:

    def __init__(self, directory_path):
        self.root_path = directory_path
        self.paths = os.listdir(self.root_path)
        self.paths = shuffle(self.paths)
        self.queue = deque(self.paths)

        self.numData = len(self.paths)
        self.dataPresented = 0

    def load_from_filepath(path):

        with open(path, 'rb') as fp: 
            face = face_pb2.Face()
            face.ParseFromString(fp.read())
            return face

    def get_data(self, n):
        self.dataPresented += n

        dataPaths = [self.queue.popLeft() for i in range(n)]
        self.queue.append(dataPaths)
        data = [self.load_from_filepath(file) for file in dataPaths]

        if (self.dataPresented - n % self.numData) + n > self.numData:
            self.paths = shuffle(self.paths)
            self.queue = deque(self.paths)

        return ([datum.image for datum in data], 
                [datum.facs for datum in data])

    def get_epoch(self):
        return (self.dataPresented / self.numData) + 1


