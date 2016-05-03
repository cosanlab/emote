import cv2


class FACDatum:

    def __init__(self, image, label_array, id):
        self.image = image
        self.label_map = {fac.fac: fac.intensity for fac in label_array}
        self.id = id

    def get_image(self):
        return self.image

    def get_labels(self):
        return self.label_map.keys()

    def get_label_pairs(self):
        return self.label_map.items()

    def intensity_for_lable(self, label):
        if self.label_map.has_key(label):
            return self.label_map[label]

    def is_zero(self):
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

    def __init__(self, fac, intensity=-1):
        self.fac = fac
        self.intensity = intensity