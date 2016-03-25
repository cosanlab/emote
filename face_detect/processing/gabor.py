import cv2
import numpy as np
from sys import argv

class GaborFilter:

    def __init__(self):
        self.filters = []

    def add_filter(self, ksize, sigma, theta, lambd, gamma, psi, ktype):

         kern = cv2.getGaborKernel((ksize, ksize), sigma, theta,lambd, gamma, psi, ktype)
         kern /= 1.5*kern.sum()
         self.filters.append(kern)

    def process(self, img):
        accum = np.zeros_like(img)
        for kern in self.filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)

        return accum



def main():
    image = cv2.imread(argv[1], cv2.IMREAD_COLOR)
    cv2.imshow('Gabor Filter', image)

if __name__ == '__main__':
    main()