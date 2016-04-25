
class FEExpresser:

    def train(self):
        raise NotImplementedError

    def predict(self, face):
        raise NotImplementedError

    def get_image_size(self):
        raise NotImplementedError