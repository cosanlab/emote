from FEExpresser import FEExpresser
import tensorflow as tf
from face_feature.LandmarkDetector import LandmarkDetector

N = 2

class FENottinghamCNN(FEExpresser):

    def __init__(self, repository, codes, intensity=False):
        self.repo = repository
        self.codes = codes
        self.hasIntensity = intensity



    def train(self):
        return None

    def predict(self, face):
        return None

    def _getFaceRegions(self, image):
        ld = LandmarkDetector()
        landmarks = ld.get_landmarks(image)


def main():
    print("Testing Nottingham framework....")


if __name__ == '__main__':
    main()