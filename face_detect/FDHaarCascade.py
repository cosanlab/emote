from FDDetector import FDDetector
from util.paths import DataLoc
import cv2
import numpy as np
import os

class FDHaarCascade(FDDetector):

    def __init__(self):

        path_finder = DataLoc()
        data_path =  path_finder.get_path('haar') + '/haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(data_path)

        FDDetector.__init__(self)

    def find(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        roi_color = None

        if len(faces) > 0:
            x,y,w,h = faces[0]
            roi_color = np.copy(frame[y:y+h, x:x+w])
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        return frame, roi_color