from FDDetector import FDDetector
from util.paths import DataLoc
import util.constants as ks
import cv2
import numpy as np

SCALE_FACTOR = 0.15

class FDHaarCascade(FDDetector):
    """Uses Haar Cascades to select a rectangular portion of an image 
        most likely to have a face.
    """

    def __init__(self):

        path_finder = DataLoc()
        data_path =  path_finder.get_path(ks.kDataHaar) + '/haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(data_path)

    def find(self, frame, grayscale=False):
        """
        Given an image, attempts to locate a face and return it.

        :param frame: Image that might have a face
        :type frame: OpenCV Mat
        :param grayscale: If True, returns face in grayscale
        :type grayscale: bool

        :returns: Mat -- The detected face. Is None if no face is found 

        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        roi_color = None

        if len(faces) > 0:
            x,y,w,h = faces[0]

            roi_color = np.copy(frame[y:y+h, x:x+w])
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            if grayscale:
                roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

        return frame, roi_color
