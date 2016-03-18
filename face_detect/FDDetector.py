
#This is a base interface class for a facial detector
#All subclasses need to override methods that raise a 
#NotImplementedError
 
class FDDetector:

    def __init__(self):
        self.x = 1 #Place holder

    #required override
    def find(self, frame):
        raise NotImplementedError()

