import numpy as np
import cv2
import os.path

def process_from_file(detector, expresser, filename):

    if filename is None and os.path.isfile(filename):
        print("Invalid file provided")
        return False

    cap = cv2.VideoCapture(filename)
    
    if cap is None:
        print("Unable to create video interface")
        return False

    _process_from_video_obj(cap, detector, expresser)

    return True

def process_from_capture(detector, expresser, camera=0):

    cap = cv2.VideoCapture(camera)
    if cap is None:
        print("Unable to find capture media")
        return False

    _process_from_video_obj(cap, detector, expresser)

    return True
    

def _process_from_video_obj(capture, detector, expresser):
    while True:
        # Capture frame-by-frame
        ret, frame = capture.read()

        if ret:
            new_frame, face = detector.find(frame)
            if face is not None:
                data = expresser.predict(face)
                #Display data here
                
            # Display the resulting frame
            cv2.imshow('frame', new_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()

