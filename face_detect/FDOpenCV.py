from FDDetector import FDDetector


class FDOpenCV(FDDetector):

    def find(self, frame):
        return frame, None