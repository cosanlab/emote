from FEExpresser import FEExpresser
from FESingleAUCNN import FESingleAUCNN
import tensorflow as tf
import thread

class FEAggregatedCNN(FEExpresser):
    """Uses multiple individually trained FESingleAUCNNs together to produce predictions for a range of action units. 
    """

    def __init__(self, image_size, codes, repo=None):
        """
        :param image_size: The rectangular length the model should expect
        :type image_size: int
        :param codes: List of action units to train on and recognize
        :type codes: [int]
        :param repo: The data repository to pull training examples from. Not necessary to provide on pre-trained models
        :type repo: data.repositories.FACRepository
        """

        self.image_size = image_size
        self.codes = codes
        self.repo = repo
        self.cnns = {}
        self.isTrained = False

        for code in self.codes:
            cnn = FESingleAUCNN(self.image_size, code, repo)
            self.cnns[code] = cnn

    def predict(self, face):
        """Runs a prediction on each FESingleAUCNN to recognize each AU, then aggregates them into one list

        :param face: The face image to predict the AUs of
        :type face: OpenCV Mat or numpy ndarray

        :returns: [int] - list of predictions 
        """

        prediction = []

        for code in self.codes:
            cnn = self.cnns[code]
            prediction.append(cnn.predict(face))

        return prediction

    def train(self):
        """Trains each SingleAUCNN
        """
        for au, cnn in self.cnns.items():
            cnn.train()
            self.repo.reset_repo_for_au(au)

        self.isTrained = True

    def get_image_size(self):
        """ Returns the size of the rectangular image the model is setup to handle
        :returns: int - if returns n, the image has dimensions n x n
        """
        return self.image_size

    def __str__(self):
        return str(self.cnns)

def main():
    model = FEAggregatedCNN(96, [1,2,4])
    print(model)

if __name__ == '__main__':
    main()