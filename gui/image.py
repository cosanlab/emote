import os
import cv2
from util.constants import FRAME_WIDTH, FRAME_HEIGHT

def process_image(detector, expresser, path, out_file=None):
    if path is None or not os.path.isfile(path):
        print("Unable to find file")
        return False

    image = cv2.imread(path, cv2.IMREAD_COLOR)

    new_frame, face = detector.find(image)
    if face is not None:
        #Scale to model's desired size
        normalized_face = cv2.resize(face, (expresser.get_image_size(), expresser.get_image_size()), cv2.INTER_CUBIC)

        data = expresser.predict(normalized_face)
        print(data)

        if out_file:
            write_image_to_file(normalized_face, out_file)
        else:
            while True:
                cv2.imshow('image', new_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


    cv2.destroyAllWindows()


def write_image_to_file(img, path):
    if path is not None:
        cv2.imwrite(path, img)
    else:
        print("Output file path is invalid")