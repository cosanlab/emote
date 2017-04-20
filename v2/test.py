import os, sys
import argparse

from keras import backend as K
from keras.models import Sequential, load_model

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.externals import joblib

import itertools
import numpy as np
import matplotlib.pyplot as plt

from error import multilabel_error
from difsa import difas_repo

FACS = ['1', '2', '4', '5', '6', '9', '12', '15', '17', '20', '25', '26']
CM_OUTPUT = 'confusion.pdf'

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('model', type=str, help='Path to trained model')
    parser.add_argument('classifier', type=str, help='Path to trained classifier')
    parser.add_argument('testing_data', dest='data', help='Path to testing directory')

    return parser.parse_args()

def load_model(path):
    if not os.path.isfile(path):
        print("Invalid model path: %s" % path)
        exit(1)

    return load_model(model_path, custom_objects={'multilabel_error': multilabel_error})

def load_classifier(path):
    if not os.path.isfile(path):
        print("Invalid classifier path: %s" % path)
        exit(1)

    return joblib.loads(path)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def test_model(model, classifier, training_data_path):
    testing_repo = difsa_repo(training_data_path, eager=ALL)

    images, labels = testing_repo.get_dataset()
    net_output = model.predict(images)
    predictions = classifier.predict(net_output)

    f1 = f1_score(labels, predictions)
    print("F1 score: %s" % str(f1))

    accuracy = accuracy(lables, predictions)
    print("Accuracy: %s" % str(accuracy))

    cm = confusion_matrix(labels, predictions)
    plot_confusion_matrix(cm, FACS,
                              normalize=False,
                              title='FACs Confusion')
    plt.savefig(CM_OUTPUT)
    classification_report(labels, predictions, FACS)


def test():
    args = parse_args()
    model = load_model(args.model)
    classifier = load_classifier(args.classifier)

    test_model(model, classifier, args.data)


if __name__ == '__main__':
    test()
