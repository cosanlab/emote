import cv2

class FACDatum:
    """A structure to hold an image and its corresponding AUs
    """

    def __init__(self, image, label_array, id):
        """
            :param image: Image representation
            :type image: OpenCV Mat or numpy ndarray
            :param label_array: List of FACLabel objects assocaited with the image
            :type label_array: [FACDatum]
            :param id: a unique identifier for the datum
            :type id: whatever you want as long as it has an __eq__() defined
        """

        self.image = image
        self.label_map = {fac.fac: fac.intensity for fac in label_array}
        self.id = id

    def get_image(self):
        """ Retrieves the associated image
        
        :returns: (OpenCV Mat) Associated image
        """
        return self.image

    def get_labels(self):
        """ Retieves the AUs associated with the datum
        
        :returns: ([int]) list of AUs
        """
        return self.label_map.keys()

    def get_label_pairs(self):
        """ Gets associated AUs and their respective intensities
            
        :returns: ([(int, float)]) list of tuples as (AU, intensity)
        """
        return self.label_map.items()

    def has_au(self, au):
        """ Checks if FACDatum has an AU with intensity greater than or equal to 2

        :returns: (bool) If AU is >= than 2
        """
        return self.label_map.has_key(au) and self.label_map[au] >= 2

    def intensity_for_label(self, label):
        """ Given an AU, returns the intensity of it

        :returns: intensity of AU, 0 if the AU isn't found
        """
        if self.label_map.has_key(label):
            return self.label_map[label]

        return 0

    def is_zero(self):
        """Checks if image occurence of any AUs
            
        :returns: (bool) False if any intensity > 0, True otherwise
        """
        for fac, intensity in self.label_map.items():
            if intensity > 0:
                return False
        
        return True

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
            and self.id == other.id)

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return "ID: %s \nLabels: " % (self.id) + str(self.label_map)


class FACLabel:
    """ Simple object for AU and intensity pairs
    """

    def __init__(self, au, intensity=-1):
        """
        :param au: AU to hold
        :type au: int
        :param intensity: Possible intensity. -1 if not specified
        :type intensity: float or int preferably
        """
        self.fac = fac
        self.intensity = intensity