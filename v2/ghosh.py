#! /usr/bin/python
import sys
import logging

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D
from keras.layers.core import Dropout, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
import numpy as np


import matplotlib
matplotlib.use('Agg')   
from matplotlib import pyplot as plt

from difsa import difsa_repo

TRAINING_DIR = sys.argv[1]
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

logging.basicConfig(filename='ghosh.log', level=logging.DEBUG, filemode='w')
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


def multilabel_error(label, output):
    print("Label:  %s" % str(label))
    print("Output: %s" % str(output))

    p_hat = K.exp(output) / K.sum(K.exp(output))
    return - K.mean(K.sum(label * K.log(p_hat)))

def fbeta_score(y_true, y_pred, beta=1):

    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Weight precision and recall together as a single scalar.
    beta2 = beta ** 2
    f_score = (1 + beta2) * (precision * recall) / (beta2 * precision + recall)
    return f_score

def make_plot(data, x_label, file):

    xs = range(1,len(data)+1)
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

model.compile(optimizer, multilabel_error, metrics=['accuracy', fbeta_score])

training_repo = difsa_repo(TRAINING_DIR)
losses = MetricAccumulator('Loss')
accuracy = MetricAccumulator('Accuracy')
f1 = MetricAccumulator('F1 Score')

epoch = 1

try:
    while training_repo.get_epoch() < EPOCHS:

        if epoch != training_repo.get_epoch():
            epoch = training_repo.get_epoch()

        images, facs = training_repo.get_data(BATCH_SIZE)
        images = np.asarray(images)
        facs = np.asarray(facs)
        loss = model.train_on_batch(np.asarray(images), np.asarray(facs))
        log.info('Trainging info: loss = %s, epoch = %s' % (loss, epoch))

        losses.add(loss[0], epoch)
        accuracy.add(loss[1], epoch)
        f1.add(loss[2], epoch)

    losses.flush()
    accuracy.flush()
    f1.flush()

except Exception as e:
    log.exception(e)

try:
    make_plot(losses.data, losses.name, 'ghosh_loss.png')
    make_plot(accuracy.data, accuracy.name, 'ghosh_accuracy.png')
    make_plot(f1.data, f1.name, 'ghosh_f1.png')
except Exception as e:
    log.exception(e)

