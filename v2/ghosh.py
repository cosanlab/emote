#! /usr/bin/python
import sys

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolutional2D
from keras.layers.core import Dropout, Flatten
from keras.layers.pooling import MaxPooling2D
from keras import backend as K

import numpy as np


TESTING_DIR = sys.argv[1]
IMAGE_SIZE = 96

DROPOUT = 0.5
THRESHOLD = 
KERNEL_SIZE=5
LEARNING_RATE=0.01
WEIGHT_STD = 0.3

BATCH_SIZE = 30
EPOCHS = 500

FULL_SIZE_1 = 500
FULL_SIZE_2 = 100
FULL_SIZE_3 = 12

REDUCED_IMAGE_SIZE = 3 * IMAGE_SIZE / 4.0

def multilabel_error(label, output):
    p_hat = K.exp(output) / K.sum(K.exp(output))
    return -K.mean(K.sum(label * K.log(p_hat)))

model = Sequential([
    Convolution2D(70, KERNEL_SIZE, KERNEL_SIZE, border_mode='same', input_shape=(3, IMAGE_SIZE, IMAGE_SIZE)),
    Activation('relu'),
    Dropout(DROPOUT),
    MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'),

    Convolution2D(10, KERNEL_SIZE, KERNEL_SIZE, border_mode='same', input_shape=(70, KERNEL_SIZE, KERNEL_SIZE)),
    Activation('relu'),
    Dropout(DROPOUT),
    MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'),

    Flatten(),

    Dense(FULL_SIZE_1, input_dim=REDUCED_IMAGE_SIZE),
    Activation('relu'),
    Dense(FULL_SIZE_2, input_dim=FULL_SIZE_1),
    Activation('relu'),
    Dense(FULL_SIZE_3, input_dim=FULL_SIZE_2)
])

optimizer = Adam(lr=LEARNING_RATE)

model.compile(optimizer, multilabel_error, metrics=['accuracy'])

training_repo = difsa_repo(TESTING_DIR)
losses = []

while test_repo.get_epoch() < EPOCHS:
    images, facs = test_repo.get_data(BATCH_SIZE)
    images = np.asarray(images)
    facs = np.asarray(facs)

    loss = model.train_on_batch(images, facs)
    losses.append(loss)

plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig("ghosh.png")


