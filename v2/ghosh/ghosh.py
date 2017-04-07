#! /usr/bin/python
import sys
import os
import logging
import traceback
import datetime

from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Convolution2D
from keras.layers.core import Dropout, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.callbacks import Callback
import numpy as np

from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.externals import joblib

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from difsa import difsa_repo, difsa_generator

MODEL_FILE = "ghosh_%s_%s_%s.model"
CLASS_FILE = "ghosh_class_%s_%s_%s.pkl"
ACCUR_FILE = "ghosh_acc_%s_%s_%s.png"
LOSS_FILE  = "ghosh_loss_%s_%s_%s.png"
LOG_FILE   = "ghosh_%s_%s_%s.log"

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

log = None
ALL = True

class GhoshModel:

    def __init__(self, train_dir, validation_dir,
                        learning_rate=LEARNING_RATE,
                        kernel_size=KERNEL_SIZE,
                        batch_size=BATCH_SIZE,
                        output_dir=None,
                        retrain=False):
        global log
        self.train_dir = train_dir
        self.validation_dir = validation_dir
        self.learning_rate = learning_rate
        self.kernel_size = int(kernel_size)
        self.batch_size = int(batch_size)
        self.retrain = retrain
        self.output_dir = output_dir

        #Setup logging
        if self.output_dir is not None:
            log_file = self.get_file_path(LOG_FILE)
            filepath = os.path.join(self.output_dir, self.get_file_path(LOG_FILE))

            log = logging.getLogger("ghosh_%s_%s_%s" % (self.learning_rate, self.kernel_size, self.batch_size))
            log.info("Logging to %s" % filepath)

            fmt = "%(levelname) -10s %(asctime)s %(module)s:%(lineno)s %(funcName)s %(message)s"
            handler = logging.FileHandler(filepath, mode='w')
            handler.setFormatter(logging.Formatter(fmt))
            log.addHandler(handler)
            log.setLevel(logging.DEBUG)


    def run(self):
        log.info("Loading repositories")
        training_repo = difsa_repo(self.train_dir, eager=ALL)
        validation_repo = difsa_repo(self.validation_dir, eager=ALL)

        model = self.getModel(training_repo)
        classifier = self.getClassifier()

        training_repo.reset()
        training_repo.reset_epoch()

        #Fit QDA/OneVRest
        try:
            images, facs = training_repo.get_dataset()
            predictions = model.predict(images, batch_size=self.batch_size)
            classifier.fit(predictions, facs)
        except Exception as e:
            print("ERROR: " + str(e))
            log.error(e)
            exit()

        #Save classifier
        classifier_path = os.path.join(self.output_dir, self.get_file_path(CLASS_FILE))
        log.info("Saving classifier to %s" % classifier_path)
        joblib.dump(classifier, classifier_path)
        log.info("Save classifier successfully")

        #Cross-validation test
        f1_score_acc = MetricAccumulator('F1 Score')
        accuracy_acc = MetricAccumulator('Accuracy')

        try:
            while validation_repo.get_epoch() == 1:
                images, facs = validation_repo.get_data(BATCH_SIZE*5)
                nn_predictions = model.predict(images, batch_size=self.batch_size)
                class_preds = classifier.predict(nn_predictions)

                f1_score_acc.add(f1_score(facs, class_preds, average=None), 1)
                accuracy_acc.add(accuracy_score(facs, class_preds, normalize=True), 1)

            f1_score_acc.flush()
            accuracy_acc.flush()
            log.info(self.getModelParams())
            log.info('F1 Score Average: ' + str(f1_score_acc.average()))
            log.info('Accuracy: ' + str(accuracy_acc.average()))

        except Exception as e:
            traceback.print_exc()
            log.error(e)

        return f1_score_acc.average(), accuracy_acc.average()

    def getModel(self, training_repo):
        model = None
        model_path = os.path.join(self.output_dir, self.get_file_path(MODEL_FILE))

        try:
            if os.path.isfile(model_path) and not self.retrain:
                log.info("Loading model from file %s" % model_path)
                model = load_model(model_path, custom_objects={'multilabel_error': multilabel_error})
            else:
                model = self.train_model(training_repo)

                if model is not None:
                    model.save(model_path)
                    log.info("Saving model to %s" % model_path)

        except Exception as e:
            log.error(e)
            exit()

        return model

    def getClassifier(self):
        classifier = None
        class_path = os.path.join(self.output_dir, self.get_file_path(CLASS_FILE))

        try:
            if os.path.isfile(class_path) and not self.retrain:
                log.info("Loading classifier from %s" % class_path)
                classifier = joblib.loads(class_path)
            else:
                classifier = OneVsRestClassifier(qda(store_covariances=True))
        except Exception as e:
            log.error(e)
            exit()

        return classifier

    def train_model(self, training_repo):
        model = Sequential([
            Convolution2D(70, self.kernel_size, self.kernel_size, border_mode='same', input_shape=(1, IMAGE_SIZE, IMAGE_SIZE)),
            LeakyReLU(EPSILON),
            Dropout(DROPOUT),
            MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'),

            Convolution2D(10, self.kernel_size, self.kernel_size, border_mode='same', input_shape=(70, self.kernel_size, self.kernel_size)),
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

        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss=multilabel_error, metrics=['accuracy'])

        losses = MetricAccumulator('Loss')
        accuracy = MetricAccumulator('Accuracy')

        training_repo.reset()
        training_repo.reset_epoch()

        if ALL:
            images, facs = training_repo.get_dataset()
            history = model.fit(images, facs, nb_epoch=EPOCHS, verbose=2, callbacks=[LossHistory()])

            return model
        try:
            #Training loop
            while training_repo.get_epoch() <= EPOCHS:

                images, facs = training_repo.get_data(self.batch_size)
                images = np.asarray(images)
                facs = np.asarray(facs)

                if images.size == 0 or facs.size == 0:
                    log.info('Repo returned an empty array')
                    continue

                loss = model.train_on_batch(np.asarray(images), np.asarray(facs))
                log.info('Training info: loss = %s, epoch = %s' % (loss, training_repo.get_epoch()))

                losses.add(loss[0], training_repo.get_epoch())
                accuracy.add(loss[1], training_repo.get_epoch())

            losses.flush()
            accuracy.flush()

            acc_path = os.path.join(self.output_dir, self.get_file_path(ACCUR_FILE))
            loss_path = os.path.join(self.output_dir, self.get_file_path(LOSS_FILE))

            try:
                make_plot(losses.data, losses.name, loss_path)
                make_plot(accuracy.data, accuracy.name, loss_path)
            except Exception as e:
                log.exception(e)

            return model

        except Exception as e:
            log.exception(e)

    def get_file_path(self, base):
        return base % (str(self.batch_size), str(self.kernel_size), str(self.learning_rate))

    def getModelParams(self):
        return '---PARAMETERS---\n' + \
               'Learning rate: ' + str(self.learning_rate) + '\n' + \
               '      Dropout: ' + str(DROPOUT) + '\n' + \
               '   Batch size: ' + str(self.batch_size) + '\n' + \
               '       Epochs: ' + str(EPOCHS) + '\n' + \
               '  Kernel Size: ' + str(self.kernel_size)

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        log.info(logs.get('loss'))
        self.losses.append(logs.get('loss'))

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


def multilabel_error(label, output):
    p_hat = K.exp(output) / K.sum(K.exp(output))
    return -K.mean(K.sum(label * K.log(p_hat)))


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

if __name__ == '__main__':
    training = sys.argv[1]
    validation = sys.argv[2]
    log_dir = sys.argv[3]
    ghosh = GhoshModel(training, validation, LEARNING_RATE, KERNEL_SIZE, BATCH_SIZE, log_dir)
    f1 = ghosh.run()
    print("Ghosh test model: F1_Score = %s" % str(f1))
