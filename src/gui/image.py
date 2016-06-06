import os
import cv2
from util.constants import FRAME_WIDTH, FRAME_HEIGHT

def process_image(detector, expresser, path, out_file=None):
    """ Uses a detector model, expresser model, and prints the results of the expresser model for the given image path

    :param detector: Model for detecting a face
    :type detector: FDDetector
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
    new_frame, face = detector.find(image)

    #If it doesn't exist, exit
    if face is not None:
        #Scale to model's desired size
        normalized_face = cv2.resize(face, (expresser.get_image_size(), expresser.get_image_size()), cv2.INTER_CUBIC)

        #Run recognition on normalized face
        data = expresser.predict(normalized_face)
        print(data)

        if out_file: #If out_file is provided, write normalized face to file
            write_image_to_file(normalized_face, out_file)
        else:
            while True:
                cv2.imshow('image', new_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


    cv2.destroyAllWindows()

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