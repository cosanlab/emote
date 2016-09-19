
import sys
import os
import random
import csv
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import emote_data


FAC_DATA = 'fac_data'
IMAGE_DIR = 'Images'
AU_DIR = 'AUs'
BUFFER_SIZE = 50
TRAINING_PERCENT = 70
TESTING_PERCENT = (100 - TRAINING_PERCENT) / 2.0
VALIDATION_PERCENT = (100 - TRAINING_PERCENT) / 2.0
AUS = [1,2,4,5,6,9,12,15,17,20,25,26]

TRAINING_DIR = 'training'
TESTING_DIR = 'testing'
VALIDATION_DIR = 'validation'

def write_records(root_dir, dest_dir):

    print("Beginning writing")

    #Images divided into subjects, so loop over those dirs
    images_dir = os.path.join(root_dir, IMAGE_DIR)
    if not os.path.isdir(images_dir):
        raise ValueError("There is no 'Images' directory in the specified dataset directory")

    for subj in sorted(os.listdir(images_dir)):

        print("Processing subject:%s" % subj)
        #Subject image dirs have same name as csv files...
        subj_image_dir = os.path.join(images_dir, subj)
        subject_csv_file = os.path.join(root_dir, AU_DIR, subj)
        subject_ptr = open(subject_csv_file)

        #Read in the CSV
        labels = _get_labels_for_subject(subject_ptr)

        #then for every image per subject read it in
        for i, filename in enumerate(sorted(os.listdir(subj_image_dir))):
            print("Processing %s" % filename)
            #Load the image
            file_path = os.path.join(subj_image_dir, filename)
            frame = int(filename.split('_')[2][:-4])

            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            location = TRAINING_DIR
            rand = random.random() * 100
            if rand > TRAINING_PERCENT + VALIDATION_PERCENT:
                location = VALIDATION_DIR
            elif rand > TRAINING_PERCENT:
                location = TESTING_DIR

            #Write to tfrecord
            record_path = os.path.join(dest_dir, location, filename + '.tfrecord')
            print("Writing %s" % record_path)
            faceBuffer = emote_data.Face()
            faceBuffer.facs = labels[i]
            faceBuffer.image = image.tostring()
            fp = open(record_path, 'wb')
            fp.write(faceBuffer.SerializeToString())
            fp.close()

def _get_labels_for_subject(file_ptr):
    """ Reads the CSV file and organizes it into a list

    :param file_ptr: pointer to a subject's csv
    :type file_ptr: file pointer
    :returns: list of FACLabel objects
    """
    labels = []

    #Open CSC
    reader = csv.reader(file_ptr, delimiter=',')

    for i, row in enumerate(reader):
        facs = []
        for j, intensity in enumerate(row):
            if j == 0:
                continue

            facs.append(int(intensity))

        labels.append(facs)
        
    return labels


def main():
    # params: 
    #   root_dir: The root directory of the dataset to be converted
    #   dest_dir: The destination directory for the dataset
    root_dir = sys.argv[1]
    dest_dir = sys.argv[2]

    write_records(root_dir, dest_dir)


if __name__ == '__main__':
    main()