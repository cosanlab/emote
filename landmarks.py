import sys
import os
import dlib
import glob
import cv2

predictor_path = sys.argv[1]
face_image_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

img = cv2.imread(face_image_path)
dets = detector(img, 1)
shape = predictor(img, dets)

print(shape)