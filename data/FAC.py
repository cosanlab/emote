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

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
            and self.id == other.id)

    def __hash__(self):
        return hash(self.id)


class FACLabel:

    def __init__(self, fac, intensity=-1):
        self.fac = fac
        self.intensity = intensity