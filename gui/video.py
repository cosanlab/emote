import numpy as np
import cv2
import os.path
from util.constants import FRAME_WIDTH, FRAME_HEIGHT

def process_from_file(detector, expresser, filename, rate=30, out_file=None):

    if filename is None or not os.path.isfile(filename):
        print("Invalid file provided")
        return False

    if rate is None:
        rate = 30

    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_FPS, rate)
    
    if cap is None:
        print("Unable to create video interface")
        return False

    writer = None
    if out_file is not None:
        writer = _create_video_writer_for_file(out_file, framerate=rate)

    _process_from_video_obj(cap, detector, expresser, writer)

    return True

def process_from_capture(detector, expresser, camera=0, out_file=None):

    cap = cv2.VideoCapture(camera)
    if cap is None:
        print("Unable to find capture media")
        return False

    writer = None
    if out_file is not None:
        writer = _create_video_writer_for_file(out_file)

    _process_from_video_obj(cap, detector, expresser, writer, show_window=True)

    return True

def process_video_into_frames(detector, expresser, filename, out_dir, rate=30):

    if filename is None or not os.path.isfile(filename):
        print("Invalid file provided")
        return False

    if out_dir is None or not os.path.isdir(out_dir):
        print("Invalid directory provided for output")
        return False

    if rate is None:
        rate = 30

    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_FPS, rate)

    writer = FrameWriter(out_dir, filename)

    _process_from_video_obj(cap, detector, expresser, writer)

    return True

def _process_from_video_obj(capture, detector, expresser, writer=None, show_window=False):
    while True:
        # Capture frame-by-frame
        ret, frame = capture.read()

        if ret:
            new_frame, face = detector.find(frame)
            if face is not None:

                normalized_face = cv2.resize(face, (FRAME_WIDTH, FRAME_WIDTH), cv2.INTER_CUBIC)
                data = expresser.predict(normalized_face)
                #Display data here

                if writer is not None:
                    writer.write(normalized_face)

            if show_window:
                # Display the resulting frame
                cv2.imshow('frame', new_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()

def _create_video_writer_for_file(path, framerate=30):
    if path is None:
        return None

    writer = cv2.VideoWriter(filename=path, fourcc=cv2.VideoWriter_fourcc(*'DIVX'), fps=framerate, frameSize=(FRAME_WIDTH, FRAME_HEIGHT))

    return writer


class FrameWriter:
    def __init__(self, out_dir, filename):
        self.out_dir = out_dir
        self.filename = os.path.splitext(os.path.split(filename)[1])[0]
        self.frame_count = 1

    def write(self, image):

        out_file = os.path.normpath(self.out_dir + '/' + self.filename + '_' + str(self.frame_count) + '.png')
        print("Writing to " + out_file)
        cv2.imwrite(out_file, image)
        self.frame_count += 1
