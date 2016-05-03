from FEExpresser import FEExpresser
from FESingleAUCNN import FESingleAUCNN
import tensorflow as tf

class FEAggregatedCNN(FEExpresser):

    def __init__(self, image_size, codes, repo=None):
        self.image_size = image_size
        self.codes = codes
        self.repo = repo
        self.cnns = {}

        for code in self.codes:
            cnn = FESingleAUCNN(self.image_size, code, repo)
            self.cnns[code] = cnn

    def predict(self, face):

        prediction = []

        for code in self.codes:
            cnn = self.cnns[code]
            prediction.append(cnn.predict(face))

        return prediction

    def train(self):
        for cnn in self.cnns.values():
            cnn.train()
            self.repo.reset_repo()

    def get_image_size(self):
        return self.image_size

    def __str__(self):
        return str(self.cnns)

def main():
    model = FEAggregatedCNN(96, [1,2,4])
    print(model)

if __name__ == '__main__':
    main()