#! /usr/bin/python
import sys
import os
import logging
import traceback

from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Convolution2D
from keras.layers.core import Dropout, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
import numpy as np

from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer


import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from difsa import difsa_repo

TRAINING_DIR = sys.argv[1]
VALIDATION_DIR = sys.argv[2]
MODEL_PATH = sys.argv[3]

IMAGE_SIZE = 96

DROPOUT = 0.5
KERNEL_SIZE = 5
LEARNING_RATE = 0.0001
WEIGHT_STD = 0.3

BATCH_SIZE = 64
EPOCHS = 500

FULL_SIZE_1 = 500
FULL_SIZE_2 = 100
FULL_SIZE_3 = 12

REDUCED_IMAGE_SIZE = IMAGE_SIZE / 4.0

EPSILON = 0.0001

format = "%(levelname) -10s %(asctime)s %(module)s:%(lineno)s %(funcName)s %(message)s"

logging.basicConfig(filename='ghosh.log', level=logging.DEBUG, filemode='w', format=format)
log = logging.getLogger('GHOSH')


class MetricAccumulator:

    def __init__(self, name):
        self.name = name
        self.current_epoch = 1
        self.data = []
        self.total = 0
        self.count = 0

    def add(self, data, epoch):
        if epoch > self.current_epoch:
            self.current_epoch = epoch
            self.data.append(self.total / self.count)

            self.count = 0
            self.total = 0

        self.total += data
        self.count += 1

    def flush(self):
        self.data.append(self.total / self.count)
        self.count = 0
        self.total = 0

    def average(self):
        return sum(self.data) / float(len(self.data))


def getModelParams():
    return '---PARAMETERS---\n' + \
           'Learning rate: ' + str(LEARNING_RATE) + '\n' + \
           '      Dropout: ' + str(DROPOUT) + '\n' + \
           '   Batch size: ' + str(BATCH_SIZE) + '\n' + \
           '       Epochs: ' + str(EPOCHS) + '\n' + \
           '  Kernel Size: ' + str(KERNEL_SIZE)


def multilabel_error(label, output):
    print("Label:  %s" % str(label))
    print("Output: %s" % str(output))

    p_hat = K.exp(output) / K.sum(K.exp(output))
    return - K.mean(K.sum(label * K.log(p_hat)))


def make_plot(data, x_label, file):

    xs = range(1, len(data)+1)
    ys = data

    fit = np.polyfit(xs, ys, deg=1)
    polynomial = np.poly1d(fit)
    y_fit = polynomial(xs)

    plt.scatter(xs, data)
    plt.plot(xs, y_fit, color='red')
    plt.xlabel('Epochs')
    plt.ylabel(x_label)
    plt.grid(True)
    plt.savefig(file)
    plt.close()


def ndarray_of_lists(ndarray2d):
    length = len(ndarray2d)
    new_array = np.empty((length,))

    for i, array in enumerate(ndarray2d):
        new_array[i] = array.tolist()

    return new_array


def train_model(training_repo):
    model = Sequential([
        Convolution2D(70, KERNEL_SIZE, KERNEL_SIZE, border_mode='same', input_shape=(1, IMAGE_SIZE, IMAGE_SIZE)),
        LeakyReLU(EPSILON),
        Dropout(DROPOUT),
        MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'),

        Convolution2D(10, KERNEL_SIZE, KERNEL_SIZE, border_mode='same', input_shape=(70, KERNEL_SIZE, KERNEL_SIZE)),
        LeakyReLU(EPSILON),
        Dropout(DROPOUT),
        MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'),

        Flatten(),

        Dense(FULL_SIZE_1, input_dim=REDUCED_IMAGE_SIZE),
        LeakyReLU(EPSILON),
        Dense(FULL_SIZE_2, input_dim=FULL_SIZE_1),
        LeakyReLU(EPSILON),
        Dense(FULL_SIZE_3, input_dim=FULL_SIZE_2),
        LeakyReLU(EPSILON)
    ])

    optimizer = Adam(lr=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=multilabel_error, metrics=['accuracy'])

    losses = MetricAccumulator('Loss')
    accuracy = MetricAccumulator('Accuracy')

    training_repo.reset()
    training_repo.reset_epoch()

    try:
        #Training loop
        while training_repo.get_epoch() <= EPOCHS:

            images, facs = training_repo.get_data(BATCH_SIZE)
            images = np.asarray(images)
            facs = np.asarray(facs)
            loss = model.train_on_batch(np.asarray(images), np.asarray(facs))
            log.info('Training info: loss = %s, epoch = %s' % (loss, training_repo.get_epoch()))

            losses.add(loss[0], training_repo.get_epoch())
            accuracy.add(loss[1], training_repo.get_epoch())

        losses.flush()
        accuracy.flush()

        try:
            make_plot(losses.data, losses.name, 'ghosh_loss.png')
            make_plot(accuracy.data, accuracy.name, 'ghosh_accuracy.png')
        except Exception as e:
            log.exception(e)

        return model

    except Exception as e:
        log.exception(e)

model = None
training_repo = difsa_repo(TRAINING_DIR)
validation_repo = difsa_repo(VALIDATION_DIR)

try:
    if os.path.isfile(MODEL_PATH):
        log.info("Loading model from file %s" % MODEL_PATH)
        model = load_model(MODEL_PATH, custom_objects={'multilabel_error': multilabel_error})
    else:
        model = train_model(training_repo)
        model.save(MODEL_PATH)
        log.info("Saving model to %s" % MODEL_PATH)
except Exception as e:
    log.error(e)

priors = training_repo.get_dataset_priors()
print(priors)
classifier = OneVsRestClassifier(qda(store_covariances=True))

training_repo.reset()
training_repo.reset_epoch()

try:
    images, facs = training_repo.get_dataset()
    print(images.shape)
    print(facs.shape)
    predictions = model.predict(images, batch_size=BATCH_SIZE)
    classifier.fit(predictions, facs)
except Exception as e:
    print("ERROR: " + str(e))
    log.error(e)
    exit()


#Cross-validation test
f1_score_acc = MetricAccumulator('F1 Score')
accuracy_acc = MetricAccumulator('Accuracy')

try:
    while validation_repo.get_epoch() == 1:
        images, facs = validation_repo.get_data(BATCH_SIZE*5)
        nn_predictions = model.predict(images, batch_size=BATCH_SIZE)
        class_preds = classifier.predict(nn_predictions)
        print(class_preds)
        f1_score_acc.add(f1_score(facs, class_preds, average=None), 1)
        accuracy_acc.add(accuracy_score(facs, class_preds, normalize=True), 1)

    f1_score_acc.flush()
    accuracy_acc.flush()
    log.info(getModelParams())
    log.info('F1 Score Average: ' + str(f1_score_acc.average()))
    log.info('Accuracy: ' + str(accuracy_acc.average()))
except Exception as e:
    traceback.print_exc()
    print(e)
    log.error(e)
