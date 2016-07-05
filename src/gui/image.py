import os
import cv2
from util.constants import FRAME_WIDTH, FRAME_HEIGHT
from face_process import detect_and_align_face

def process_image(expresser, path, grayscale=False):
    """ Uses a detector model, expresser model, and prints the results of the expresser model for the given image path

    :param expresser: Model for recognizing expressions
    :type expresser: FEExpresser
    :param path: Path to image to be processed
    :type path: str
    :param out_file: Path to write results of detector to
    :type out_file: str
    """

    if path is None or not os.path.isfile(path):
        print("Unable to find file")
        return False

    image = cv2.imread(path, cv2.IMREAD_COLOR)

    #Find the face in the provided image
    face = detect_and_align_face(image, expresser.get_image_size(), grayscale)

    #If it doesn't exist, exit
    if face is not None:
        #Run recognition on normalized face
        data = expresser.predict(face)
        print(data)
        write_image_to_file(face, out_file)


def write_image_to_file(img, path):
    """ Writes image representation to some path

    :param img: Representation of image to write
    :type img: OpenCV Mat or numpy ndarray
    :param path: Path to write image to
    :type path: str
    """
    if path is not None:
        cv2.imwrite(path, img)
    else:
        print("Output file path is invalid")