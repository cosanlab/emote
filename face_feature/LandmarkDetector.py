import os
import dlib
from skimage import io
from util.paths import DataLoc

class LandmarkDetector:

    def __init__(self):
        path_finder = DataLoc()
        self.predictor_path = os.path.normpath(path_finder.get_path('landmarks') + "/shape_predictor_68_face_landmarks.dat")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(self.predictor_path))

    def get_landmarks(self, image):
        dets = dlib.rectangle(0, 0, image.shape[0], image.shape[1])
        shape = self.predictor(image, dets)
        return shape

def main():
    detector = LandmarkDetector()
    img = io.imread("../data/test_data/single_processed.jpg")
    dets = detector.get_landmarks(img)
    print(dets)

if "__main__" == __name__:
    main()