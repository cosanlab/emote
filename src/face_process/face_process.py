import openface
import cv2
import sys, os
from util import paths

DLIB_PREDICTOR = os.path.join(paths.get_project_home(), 'src', 'data', 'facial_landmarks', 'shape_predictor_68_face_landmarks.dat')

def detect_and_align_face(face, image_size, grayscale=False):
    align = openface.AlignDlib(DLIB_PREDICTOR)
    bb = align.getLargestFaceBoundingBox(face)
    alignedFace = align.align_v2(image_size, face, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

    return cv2.cvtColor(alignedFace, cv2.COLOR_BGR2GRAY)


if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    img = detect_and_align_face(img, 96, True)
    cv2.imwrite(sys.argv[2], img)