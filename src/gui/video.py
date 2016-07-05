import numpy as np
import cv2
import os.path
from util.constants import FRAME_WIDTH, FRAME_HEIGHT

def process_from_file(expresser, filename, rate=30, outfile=None):
    """Processes frames of a video files. Detects faces and recognizes expressions by frame and outputs expression results
    
    :param expresser: Model to recognize facial expressions
    :type expresser: face_express.FEExpresser
    :param filename: Path to video file
    :type filename: str
    :param out_file: Path to write processed frames to video file
    :type out_file: str

    :returns: (bool) True if successfully able to process file
    """

    if filename is None or not os.path.isfile(filename):
        print("Invalid file provided")
        return False

    #Open video file and set framerate
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_FPS, rate)
    
    if cap is None:
        print("Unable to create video interface")
        return False

    #Setup writer if necessary
    writer = None
    if out_file is not None:
        writer = _create_video_writer_for_file(out_file, framerate=rate)

    #Send all setup objects to processing loop
    _process_from_video_obj(cap, expresser, writer)

    return True

def process_from_capture(expresser, camera=0, out_file=None):
    """ Process frames from live video. This probably only works on a machine with a GUI. Depends on how OpenCV was built
    
    :param expresser: Model to recognize facial expressions
    :type expresser: face_express.FEExpresser
    :param camera: ID for camera, default is usally 0
    :type camera: int
    :param out_file: Path to write processed frames to video file
    :type out_file: str
    """

    # Open camera
    cap = cv2.VideoCapture(camera)
    if cap is None:
        print("Unable to find capture media")
        return False

    #Create writer object
    writer = None
    if out_file is not None:
        writer = _create_video_writer_for_file(out_file)

    #Pass everything to main processing loop
    _process_from_video_obj(cap, expresser, writer)

    return True


def process_video_into_frames(expresser, filename, out_dir, rate=30):
    """Processes frames of a video files. Detects faces and recognizes expressions by frame and outputs expression results. This will save individual frames as images. Can be used to processed dataset videos into individual frame images
    
    :param detector: Detection model to find face in video
    :type detector: face_detect.FDDetector
    :param expresser: Model to recognize facial expressions
    :type expresser: face_express.FEExpresser
    :param filename: Path to video file
    :type filename: str
    :param out_file: Path to write processed frames to video file
    :type out_file: str

    :returns: (bool) True if successfully able to process file
    """

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

    _process_from_video_obj(cap, expresser, writer)

    return True

def _process_from_video_obj(capture, expresser, writer=None):
    """ Primary video processing loop to read from either file or live (OpenCV handles them with the same object), detect the face, and run expression recognition (with expresser), will with writer object as well

    :param capture: OpenCV VideoCapture object loaded with video stream to read
    :type capture: VideoCapture
    :param expresser: Model to recognize facial expressions
    :type expresser: face_express.FEExpresser
    :param writer: Object with an .write() function. Abstraction to write however calling function may want
    :type writer: Some object with a defined .write()
    :param show_window: Set to True if you want to view processing in live window
    :type show_window: bool
    """
    while True:
        # Capture frame-by-frame
        ret, frame = capture.read()

        if ret:
            face = detect_and_align_face(frame)
            if face is not None:
                data = expresser.predict(normalized_face)
                print(data)
                if writer is not None:
                    writer.write(normalized_face)
        else:
            break

    # When everything done, release the capture
    capture.release()

def _create_video_writer_for_file(path, framerate=30):
    """Creates preconfigured OpenCV writer object which writes frames to a video file
    ToDo: I do't think this works properly....

    :param path: Path to write video to
    :type path: str
    :param framerate: Desired video framerate
    :type framerate: int

    :returns: cv2.VideoWriter
    """
    if path is None:
        return None

    writer = cv2.VideoWriter(filename=path, fourcc=cv2.VideoWriter_fourcc(*'DIVX'), fps=framerate, frameSize=(FRAME_WIDTH, FRAME_HEIGHT))

    return writer


class FrameWriter:
    """Simple class to write frames of a video to a directory with numbered names
    Frames are written in the format <VideoFileName>_<FrameNumber>.png
    """

    def __init__(self, out_dir, filename):
        """
        :param out_dir: Directory to write frames to
        :type out_dir: str
        :param filename: Name of video file
        :type filename: str
        """
        self.out_dir = out_dir
        self.filename = os.path.splitext(os.path.split(filename)[1])[0]
        self.frame_count = 1

    def write(self, image):
        """ Writes image to file, with next frame number
            
        :param image: Image representation
        :type image: OpenCV Mat
        """
        out_file = os.path.normpath(os.path.join(self.out_dir,str(self.frame_count)))
        print("Writing to " + out_file)
        cv2.imwrite(out_file, image)
        self.frame_count += 1
