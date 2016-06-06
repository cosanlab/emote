import cv2
import csv
import os
import random
from data.FAC import FACDatum, FACLabel
from data.repositories import FACRepository
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
    """Repository to manage loading the DIFSA dataset

    Internally organized by AU in order to return data points that only contain specific AUs
    """

    def __init__(self):
        path_finder = DataLoc()

        self.data_dir = path_finder.get_path(ks.kDataDIFSA)
        self.used_data = []
        self.used_data_au = {au: [] for au in AUS}
        self.training = set()
        self.training_au = {au: set() for au in AUS}
        self.testing = set()
        self.total = 0
        self._load_from_file()

        FACRepository.__init__(self, BUFFER_SIZE)

    def _load_from_file(self):
        """ Loads the complete dataset into memory because it was easier than splitting it up on command.... 
        
        ToDo: Don't load the entire dataset into memory at once....
        """

        #Images divided into subjects, so loop over those dirs
        images_dir = self.data_dir + IMAGE_DIR
        for subj in sorted(os.listdir(images_dir)):

            #Subject image dirs have same name as csv files...
            subj_image_dir = images_dir + '/' + subj
            subject_csv_file = self.data_dir + AU_DIR + '/' + subj
            subject_ptr = open(subject_csv_file)

            #Read in the CSV
            labels = self._get_labels_for_subject(subject_ptr)

            #then for every image per subject read it in
            for i, filename in enumerate(sorted(os.listdir(subj_image_dir))):

                #Load the image
                file_path = subj_image_dir + '/' + filename
                image = cv2.imread(file_path, 0)
                image = image.reshape(len(image), len(image), 1)

                #Create the FACDatum
                frame = int(filename[6:])
                datum = FACDatum(image, labels[frame-1], subj + str(frame))
                
                #Divide into training and testing
                if self.total % TESTING_PERCENT == 0:
                    self.testing.add(datum)
                    self.total += 1

                #In order to make positive/negative examples 50/50 we randomize adding negative examples to get close...
                elif not datum.is_zero() or (datum.is_zero() and random.random() > 0.99):
                    self.training.add(datum)

                    for au in AUS:
                        if datum.has_au(au) or (not datum.has_au(au) and random.random() > 0.98):
                            self.training_au[au].add(datum)

                    self.total += 1

    def _get_labels_for_subject(self, file_ptr):
        """ Reads the CSV file and organizes it into a list

        :param file_ptr: pointer to a subject's csv
        :type file_ptr: file pointer
        :returns: list of FACLabel objects
        """
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
        """Pops training data from sets already in memory

        :returns: list of FACDatum objects
        """
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
        """ Get data for specific AU

        :param n: number of data to retrieve
        :type n: int
        :param au: AU
        :type int:
        """
        data = []

        for i in xrange(n):
            try:
                datum = self.training_au[au].pop()
                data.append(datum)
                self.used_data_au[au].append(datum)
            except Exception:
                break

        return data

    def reset_repo_for_au(self, au):
        """Resets specific a specific set of AUs
        
        :param au: AU set to reset
        :type au: int
        """
        self.training_au[au] = set(self.used_data_au)
        self.used_data_au[au] = []

    def reset_repo(self):
        """Resets complete dataset to unreturned state(not including AUs)
        """
        self.training = set(self.used_data)
        self.used_data = []

    def get_testing_items(self):
        """Retrieves a list of test data
        
        :returns: [data.FACDatum]
        """
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