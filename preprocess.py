#File that doesn't have tensorflow dependency
import cv2
import sys
import argparse
from gui.video import FrameWriter
from face_detect.FDHaarCascade import FDHaarCascade

def main():

    #Parse command args
    parser = argparse.ArgumentParser(description='Emote - Preprocess: A utility to detect faces in video or images without installing tensorflow')
    parser.add_argument('command', type=str, help='Subcommand to be run')

    args = parser.parse_args(sys.argv[1:2])

    if args.command is None:
        print_help()
    elif args.command == 'image':
        image(sys.argv[2:])
    elif args.command == 'video':
        video(sys.argv[2:])
    else:
        print_help()
        quit(1)

def image(options):

    parser = argparse.ArgumentParser(description='Image Face detection')
    parser.add_argument('input_file', type=str, help='Image to be processed')
    parser.add_argument('output_file', type=str, help='Output file')
    parser.add_argument('image_size', type=int, help='Output image size')

    args = parser.parse_args(options)

    path = args.input_file
    pic = cv2.imread(path)
    detector = FDHaarCascade()
    frame, face = detector.find(pic, grayscale=True)
    face = cv2.resize(face, (args.image_size, args.image_size), cv2.INTER_CUBIC)

    cv2.imwrite(args.output_file, face)
    print("Finshed ")

def video(options):
    parser = argparse.ArgumentParser(description='Video face detection')
    parser.add_argument('input_file', type=str, help='Image to be processed')
    parser.add_argument('output_dir', type=str, help='Output directory')
    parser.add_argument('image_size', type=int, help='Size of output image')

    args = parser.parse_args(options)

    detector = FDHaarCascade()
    writer = FrameWriter(args.output_dir, args.input_file)
    cap = cv2.VideoCapture(args.input_file)
    _detect_faces_from_video(cap, detector, writer, args.image_size)

def _detect_faces_from_video(cap, detector, writer, image_size):

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            new_frame, face = detector.find(frame)
            if face is not None:

                normalized_face = cv2.resize(face, (image_size, image_size), cv2.INTER_CUBIC)
                writer.write(normalized_face)
        else:
            break

    # When everything done, release the capture
    cap.release()

def print_help():
    print("preprocess [--help] <command>")
    print("")
    print("Subcommands:")
    print("image        Source is an image file")
    print("video        Source is a video file")

    print("Use the --help option on any subcommand for option")
    print("information relating to that command")

if __name__ == "__main__":
    main()