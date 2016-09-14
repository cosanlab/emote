#! /usr/bin/python

from keras.models import Sequential
from keras.layers import Dense, Activation, Convolutional2D
from keras.layers.core import Dropout, Flatten
from keras.layers.pooling import MaxPooling2D
from keras import backend as K

IMAGE_SIZE = 96

DROPOUT = 0.5
KERNEL_SIZE=5
LEARNING_RATE=0.01
WEIGHT_STD = 0.3
EPOCHS = 500

FULL_SIZE_1 = 500
FULL_SIZE_2 = 100
FULL_SIZE_3 = 12

REDUCED_IMAGE_SIZE = IMAGE_SIZE / 4.0

def multilabel_error(label, output):
    p_hat = K.exp(output) / K.sum(tf.exp(output))
    return -K.reduce_mean(K.sum(label * K.log(p_hat)))

model = Sequential([
    Convolutional2D(70, KERNEL_SIZE, KERNEL_SIZE, border_mode='same', input_shape=(1, IMAGE_SIZE, IMAGE_SIZE))
    Activation('relu'),
    Dropout(DROPOUT),
    MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'),
    
    Convolutional2D(10, KERNEL_SIZE, KERNEL_SIZE, border_mode'same', input_shape=(70, KERNEL_SIZE, KERNEL_SIZE)),
    Activation('relu'),
    Dropout(DROPOUT),
    MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'),

    Flatten(),

    Dense(FULL_SIZE_1, input_dim=REDUCED_IMAGE_SIZE),
    Activation('relu'),
    Dense(FULL_SIZE_2, input_dim=FULL_SIZE_1),
    Activation('relu'),
    Dense(FULL_SIZE_3, input_dim=FULL_SIZE_2),
    Activation('hard_sigmoid')
])

model.input()

optimizer = keras.optimizers.Adam(lr=LEARNING_RATE)

model.compile(optimizer, multilabel_error, metrics=['accuracy'])

from keras.datasets import cifar100

(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')

model.fit(X_train, y_train)
model.evaluate(X_test, y_test)