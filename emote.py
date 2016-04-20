
import argparse
import sys, os
import gui.video as vid
import gui.image as img
from util.paths import get_real_path
from util.logger import setupLogging
from train.train import setupModel
from face_detect.FDConvNet import FDConvNet
from face_detect.FDHaarCascade import FDHaarCascade
from face_express.FESVM import FESVM
from face_express.FEDeepNet import FEDeepNet
from face_express.FENottinghamCNN import FENottinghamCNN

def main():

    if not os.path.exists(get_real_path(__file__) + "/config.json"):
        print("Error: config.json missing from source directory")
        quit(1)

    setupLogging() #Configures logging system

    #Parse command args
    parser = argparse.ArgumentParser(description='Emote: A face detection and facial expression recognition pipeline')
    parser.add_argument('command', type=str, help='Subcommand to be run')

    args = parser.parse_args(sys.argv[1:2])

    if args.command is None:
        print_help()
    elif args.command == 'live':
        live(sys.argv[2:])
    elif args.command == 'image':
        image(sys.argv[2:])
    elif args.command == 'video':
        video(sys.argv[2:])
    elif args.command == 'train':
        config = open("./config.json")
        model = setupModel(config)

        if not model.isTrained:
            model.train()
    else:
        print_help()
        quit(1)


def live(argv):

    #setup args
    parser = argparse.ArgumentParser(description="Live subcommand: Process video from a live camera feed")
    parser.add_argument('-d', '--detector', type=str, help='Face detection algorithm', choices=['haar', 'cnn'])
    parser.add_argument('-e', '--expresser', type=str, help='Facial expression train algorithm', choices=['svm', 'dbn', 'cnn'])
    parser.add_argument('-o', '--out-file', type=str, help='Path to processed media')

    args = parser.parse_args(argv)

    detector = get_detector(args.detector)
    expresser = get_expresser(args.expresser)

    vid.process_from_capture(detector, expresser, out_file=args.out_file)

def image(argv):
    parser = argparse.ArgumentParser(description="Image subcommand: process an image file")
    parser.add_argument('path',  help="The image file to be processed", type=str)
    parser.add_argument('-d', '--detector', type=str, help='Face detection algorithm', choices=['haar', 'cnn'])
    parser.add_argument('-e', '--expresser', type=str, help='Facial expression train algorithm', choices=['svm', 'dbn', 'cnn'])
    parser.add_argument('-o', '--out-file', type=str, help='Path to processed media')

    args = parser.parse_args(argv)

    detector = get_detector(args.detector)
    expresser = get_expresser(args.expresser)

    img.process_image(detector, expresser, args.path, args.out_file)

def video(argv):

    parser = argparse.ArgumentParser(description="Video subcommand: Process video from a source file")
    parser.add_argument('path',  help="The image file to be processed", type=str)
    parser.add_argument('-d', '--detector', type=str, help='Face detection algorithm', choices=['haar', 'cnn'])
    parser.add_argument('-e', '--expresser', type=str, help='Facial expression train algorithm', choices=['svm', 'dbn', 'cnn'])

    #Setup output options
    output_group = parser.add_mutually_exclusive_group(required=False)
    output_group.add_argument('-o', '--out-file', type=str, help='Path to processed video file')
    output_group.add_argument('-O', '--out-directory', type=str, help='Path to directory where processed frames will be placed')

    #Set the framerate of the video
    parser.add_argument('-r', '--frame-rate', type=int, help='Frame rate of the provided video, default is 30fps')

    args = parser.parse_args(argv)

    detector = get_detector(args.detector)
    expresser = get_expresser(args.expresser)

    if args.out_directory:
        vid.process_video_into_frames(detector, expresser, args.path,args.out_directory, args.frame_rate)

    vid.process_from_file(detector, expresser, args.path, args.frame_rate, args.out_file)

def print_help():
    print("emote [--help] <command>")
    print("")
    print("Subcommands:")
    print("live         Image source is the computer's primary camera")
    print("image        Image source is a provided image file")
    print("video        Source is a provided video file")
    print("")
    print("Face Detectors:")
    print("Select a detector using the -d or --detector option")
    print("")
    print("haar         Haar-like features and Cascading Classifiers")
    print("cnn          Uses a convolution neural net to detect faces")
    print("")
    print("Expression Recognition Models:")
    print("Select a recognition model using the -e or --expresser option")
    print("")
    print("svm          Uses an SVM approach to classify expressions")
    print("cnn          A convolution neural net approach")
    print("dbn          Utilizes deep belief neural networks")
    print("")
    print("Use the --help option on any subcommand for option")
    print("information relating to that command")

def get_expresser(option):
    express_method = None
    if option == 'svm' or option is None:
        express_method = FESVM()
    elif option == 'dbn':
        express_method = FEDeepNet()
    elif option  == 'cnn':
        express_method = FEConvNet()
    else:
        print_help()
        quit()

    return express_method

def get_detector(option):
    detect_method = None
    if option == 'haar' or option is None:
        detect_method = FDHaarCascade()
    elif option == 'cnn':
        detect_method = FDConvNet()
    else:
        print_help()
        quit()

    return detect_method

if __name__ == "__main__":
    main()
