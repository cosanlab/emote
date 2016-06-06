
class FEExpresser:
    ''' FEExpresser serves as a base class for expression detector objects. Any new expression detection models should extend this one as it is used to abstract away individual methods in the controller system (emote.py)

    Extending classes may have other parameter requirements to instaniate, but should have no other requirements to be used
    '''

    def train(self):
        '''Trains the expression recognition model 
            Must be overriden by any extendng class

            :raises: NotImplementedError
        '''
        raise NotImplementedError

    def predict(self, face):
        ''' Given a face image, predicts the corresponding expression
        
        :param face: Image which matches in size with the parameters of the model
        :type face: OpenCV Mat (numpy ndarray would also probably work)
        :returns: list of expression data
        :raises: NotImplementedError
        '''
        raise NotImplementedError

    def get_image_size(self):

        raise NotImplementedError