from FACRepository import FACRepository

import cv2
import csv
import os
from data.FAC import FACDatum, FACLabel
from data.repositories.FACRepository import FACRepository
from util.paths import DataLoc
import util.constants as ks

FAC_DATA = 'fac_data'
IMAGE_DIR = '/Images'
AU_DIR = '/AUs'
BUFFER_SIZE = 50
IMAGE_EXT = '.jpg'
TESTING_PERCENT = 15
AUS = [1,2,4,5,6,9,12,15,17,20,25,26]

class DIFSARepository(FACRepository):

    def __init__(self):
        path_finder = DataLoc()

        self.data_dir = path_finder.get_path(ks.kDataDIFSA)
        self.total = 0
        self.training = set()
        self.testing = set()

        self._load_from_file()

        FACRepository.__init__(self, BUFFER_SIZE)

    def _load_from_file(self):

        images_dir = self.data_dir + IMAGE_DIR
        for dir in sorted(os.listdir(images_dir)):
            for file in sorted(os.listdir(dir)):
                file_path = dir + '/' + file
                image = cv2.imread(file_path)
                image = image.reshape(len(image), len(image), 1)

                fac = self._get_label_for_file(file)

                datum = FACDatum(image, fac, self.total)

                if self.total % 15 == 0:
                    self.testing.add(datum)
                else:
                    self.training.add(datum)

                self.total += 1

    def _get_label_for_file(self, filename):
        facs = []

        #get info
        subject = filename[0:5]
        frame = int(filename[6:])
        subject_file = self.data_dir + AU_DIR + '/' + subject

        #Open CSV
        subject_file = open(subject_file)
        reader = csv.reader(subject_file, delimiter=',')

        for i, row in enumerate(reader):
            if i == frame:
                for j, intensity in enumerate(row):
                    if j == 0:
                        continue

                    facs.append(FACLabel(AUS[j], intensity))


        return facs

    def _load_data(self, n):
        data = []

        for i in xrange(n):
            data.append(self.training.pop())

        return data

    def reload_repo(self):
        self._load_from_file()

    def get_testing_items(self):
        return list(self.testing)