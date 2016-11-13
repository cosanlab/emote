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

from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda
from sklearn.metrics import f1_score


import matplotlib
matplotlib.use('Agg')   
from matplotlib import pyplot as plt

from difsa import difsa_repo

TRAINING_DIR = sys.argv[1]
VALIDATION_DIR = sys.argv[2]

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

def getModelParams():
    return '---PARAMETERS---\n' + \
           'Learning rate: ' + LEARNING_RATE + '\n' + \
           '      Dropout: ' + DROPOUT + '\n' + \
           '   Batch size: ' + BATCH_SIZE + '\n' + \
           '       Epochs: ' + EPOCHS + '\n' + \
           '  Kernel Size: ' + KERNEL_SIZE

def multilabel_error(label, output):
    print("Label:  %s" % str(label))
    print("Output: %s" % str(output))

    p_hat = K.exp(output) / K.sum(K.exp(output))
    return - K.mean(K.sum(label * K.log(p_hat)))

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
model.compile(optimizer=optimizer, loss=multilabel_error, metrics=['accuracy'])

classifier = OneVsRestClassifier(qda())

training_repo = difsa_repo(TRAINING_DIR)
validation_repo = difsa_repo(VALIDATION_DIR)

losses = MetricAccumulator('Loss')
accuracy = MetricAccumulator('Accuracy')
f1 = MetricAccumulator('F1 Score')

epoch = 1

try:
    #Training loop
    while training_repo.get_epoch() <= EPOCHS:

        if epoch != training_repo.get_epoch():
            epoch = training_repo.get_epoch()

        images, facs = training_repo.get_data(BATCH_SIZE)
        images = np.asarray(images)
        facs = np.asarray(facs)
        loss = model.train_on_batch(np.asarray(images), np.asarray(facs))
        log.info('Training info: loss = %s, epoch = %s' % (loss, epoch))

        losses.add(loss[0], epoch)
        accuracy.add(loss[1], epoch)

    losses.flush()
    accuracy.flush()

    #Train QDA
    while training_repo.get_epoch() == epoch:
        images, facs = training_repo.get_data(BATCH_SIZE)
        predictions = model.predict_on_batch(images)
        classifier.fit(predictions, facs)


    #Cross-validation test
    images, facs = validation_repo.get_dataset()

    nn_predictions = model.predict_on_batch(images)
    class_preds = classifier.predict(nn_predictions)
    score = f1_score(facs, class_preds)
    log.info(getModelParams())
    log.info('F1 Score: %d' % score)

except Exception as e:
    log.exception(e)

try:
    make_plot(losses.data, losses.name, 'ghosh_loss.png')
    make_plot(accuracy.data, accuracy.name, 'ghosh_accuracy.png')
except Exception as e:
    log.exception(e)

