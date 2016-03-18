
import argparse
import gui.video as vid
from face_detect.FDConvNet import FDConvNet
from face_detect.FDOpenCV import FDOpenCV
from face_express.FESVM import FESVM
from face_express.FEDeepNet import FEDeepNet
from face_express.FEConvNet import FEConvNet

def main():

    #Parse args
    parser = argparse.ArgumentParser(description='Emote: A face detection and facial expression recognition pipeline')
    parser.add_argument('source', type=str, help='Source of video to be processed', choices=['live', 'file'])
    parser.add_argument('-d', '--detect-method', type=str, help='Face detection algorithm', choices=['opencv', 'cnn'])
    parser.add_argument('-e', '--express-method', type=str, help='Facial expression processing algorithm', choices=['svm', 'dbn', 'cnn'])
    parser.add_argument('-f', '--filename', type=str, help='Video file to be processed')

    arg = parser.parse_args()

    #Holder variables to be filled by args
    detect_method = None
    express_method = None


    #Get detection method, default is opencv
    if arg.detect_method == None or arg.dete_method == 'opencv':
        detect_method = FDOpenCV()
    elif arg.detect_method == 'cnn':
        detect_method = FDConvNet()

    #Get expression method, default is svm
    if arg.express_method == None or arg.express_method == 'svm':
        express_method = FESVM()
    elif arg.express_method == 'dbn':
        express_method = FEDeepNet()
    elif arg.express_method == 'cnn':
        express_method = FEConvNet()

    if express_method == None or detect_method == None:
        #Shouldn't get here
        print("Unable to properly select processing options")
        quit()

    #Get source from args, default is live
    if arg.source == None or arg.source == 'live':
        vid.process_from_capture(detect_method, express_method)
    elif arg.source == 'file':
        if arg.filename == None:
            print("Must provide a filename with the -f option")
            parser.print_help()
            quit()

        vid.process_from_file(detect_method, express_method, arg.filename)



if __name__ == "__main__":
    main()