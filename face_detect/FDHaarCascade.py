from FDDetector import FDDetector
from util.paths import get_real_path
import cv2
import numpy as np
import os

class FDHaarCascade(FDDetector):

    def find(self, frame):
        data_loc = get_real_path(__file__) + '/../data/haarcascades/haarcascade_frontalface_default.xml'

        face_cascade = cv2.CascadeClassifier(data_loc)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        roi_color = None

        if len(faces) > 0:
            x,y,w,h = faces[0]
            roi_color = np.copy(frame[y:y+h, x:x+w])
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        return frame, roi_color