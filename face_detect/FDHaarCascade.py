from FDDetector import FDDetector
import cv2
import numpy as np

class FDHaarCascade(FDDetector):

    def find(self, frame):

        face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        roi_color = None

        if len(faces) > 0:
            x,y,w,h = faces[0]
            roi_color = np.copy(frame[y:y+h, x:x+w])
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        return frame, roi_color