class FDDetector:
    """ This is a base interface class for a facial detection model
        All subclasses need to override methods that raise a NotImplementedError
    """

    def find(self, frame):
        """
        :param frame: An image that corresponds to the type of the detection model
        :type frame: image datastructure
        :raises: NotImplementedError
        """

        raise NotImplementedError()

