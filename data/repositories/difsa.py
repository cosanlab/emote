from FACRepository import FACRepository

import cv2
import csv
import os
import random
from data.FAC import FACDatum, FACLabel
from data.repositories.FACRepository import FACRepository
from util.paths import DataLoc
import util.constants as ks
from util import config

FAC_DATA = 'fac_data'
IMAGE_DIR = '/Images'
AU_DIR = '/AUs'
BUFFER_SIZE = 50
TESTING_PERCENT = 15
AUS = [1,2,4,5,6,9,12,15,17,20,25,26]

class DIFSARepository(FACRepository):

    def __init__(self):
        path_finder = DataLoc()

        self.data_dir = path_finder.get_path(ks.kDataDIFSA)
        self.used_data = {au: [] for au in AUS}
        self.training = {au: set() for au in AUS}
        self.testing = set()
        self.total = 0
        self._load_from_file()

        FACRepository.__init__(self, BUFFER_SIZE)

    def _load_from_file(self):

        images_dir = self.data_dir + IMAGE_DIR
        for subj in sorted(os.listdir(images_dir)):

            subj_image_dir = images_dir + '/' + subj
            subject_csv_file = self.data_dir + AU_DIR + '/' + subj
            subject_ptr = open(subject_csv_file)

            labels = self._get_labels_for_subject(subject_ptr)

            for i, filename in enumerate(sorted(os.listdir(subj_image_dir))):

                file_path = subj_image_dir + '/' + filename
                image = cv2.imread(file_path, 0)
                image = image.reshape(len(image), len(image), 1)
                frame = int(filename[6:])

                datum = FACDatum(image, labels[frame-1], subj + str(frame))
                
                if self.total % TESTING_PERCENT == 0:
                    self.testing.add(datum)
                    self.total += 1

                else:
                    for au in AUS:
                        if datum.has_au(au) or random.random() > 0.6:
                            self.training[au].add(datum)

                    self.total += 1

    def _get_labels_for_subject(self, file_ptr):
        labels = []

        #Open CSC
        reader = csv.reader(file_ptr, delimiter=',')

        codes = config.get_model_info()[ks.kModelFACsKey][ks.kModelFACsCodesKey]

        for i, row in enumerate(reader):
            facs = []
            for j, intensity in enumerate(row):
                if j == 0:
                    continue

                if AUS[j-1] in codes:
                    facs.append(FACLabel(AUS[j-1], int(intensity)))

            labels.append(facs)
            
        return labels

    def _load_data(self, n):
        data = []

        for i in xrange(n):
            try:
                datum = self.training.pop()
                data.append(datum)
                self.used_data.append(datum)
            except Exception:
                break

        return data

    def get_data_for_au(self, n, au):
        data = []

        for i in xrange(n):
            try:
                datum = self.training[au].pop()
                data.append(datum)
                self.used_data[au].append(datum)
            except Exception:
                break

        return data

    def reset_repo_for_au(self, au):
        self.training[au] = set(self.used_data)
        self.used_data[au] = []

    def get_testing_items(self):
        return list(self.testing)


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