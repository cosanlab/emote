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
from matplotlib import pyplot as plt

from difsa import difsa_repo

TESTING_DIR = sys.argv[1]
IMAGE_SIZE = 96

DROPOUT = 0.5
KERNEL_SIZE = 5
LEARNING_RATE = 0.0001
WEIGHT_STD = 0.3

BATCH_SIZE = 30
EPOCHS = 500

FULL_SIZE_1 = 500
FULL_SIZE_2 = 100
FULL_SIZE_3 = 12

REDUCED_IMAGE_SIZE = IMAGE_SIZE / 4.0

EPSILON = 0.0001

logging.basicConfig(filename='ghosh.log', level=logging.DEBUG)
log = logging.getLogger(__name__)

def multilabel_error(label, output):
    print("Label:  %s" % str(label))
    print("Output: %s" % str(output))

    p_hat = K.exp(output) / K.sum(K.exp(output))
    return - K.mean(K.sum(label * K.log(p_hat)))

def make_plot(data, x_label, file):
    plt.plot(data)
    plt.xlabel('Epochs')
    plt.ylabel(xlabel)
    plt.grid(True)
    plt.savefig(file)

model = Sequential([
    Convolution2D(70, KERNEL_SIZE, KERNEL_SIZE, border_mode='same', input_shape=(1, IMAGE_SIZE, IMAGE_SIZE)),
    LeakyReLU(Epochs),
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

model.compile(optimizer, multilabel_error, metrics=['accuracy'])

training_repo = difsa_repo(TESTING_DIR)
losses = []
accuracy = []

try:
    while training_repo.get_epoch() < EPOCHS:
        images, facs = training_repo.get_data(BATCH_SIZE)
        images = np.asarray(images)
        facs = np.asarray(facs)
        loss = model.train_on_batch(np.asarray(images), np.asarray(facs))
        log.info('Trainging info: loss = %s, epoch = %s' % (loss, training_repo.get_epoch()))
        losses.append(loss[0])
        accuracy.append(loss[1])
except Exception as e:
    log.error(e)

try:
    make_plot(losses, 'Loss', 'ghosh_loss.png')
    make_plot(accuracy, 'Accuracy', 'ghosh_accuracy.png')
except Exception as e:
    log.error(e)




