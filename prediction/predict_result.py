import os
import cv2
import numpy as np
import tensorflow as tf
import sys

sys.path.insert(0, 'data_preprocessing')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from pre_processing import preprocess

yo = tf.keras.models.load_model(
    'model_checkpoints/weights-improvement-06-0.633729.hdf5')


def load_model(a, b, c):

    g = yo.predict([a, b, c])
    y = g[0]
    conclusion = []
    if (y[0] > 0.5):
        conclusion.append('epidural')
    if (y[1] > 0.5):
        conclusion.append('intraparenchymal')
    if (y[2] > 0.5):
        conclusion.append('intraventricular')
    if (y[3] > 0.5):
        conclusion.append('subarachnoid')
    if (y[4] > 0.5):
        conclusion.append('subdural')
    if (len(conclusion) == 0):
        conclusion.append('Healthy')
    print(y)
    return conclusion


def feed_model():
    filename = 'data/image/' + 'bone' + '.png'
    X_train = []
    image = cv2.imread(filename=filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    X_train.append(image)
    X_train = np.array(X_train) / 255.0
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],
                              X_train.shape[2], 1)

    filenames = 'data/image/' + 'brain' + '.png'
    X_trains = []
    image = cv2.imread(filenames)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    X_trains.append(image)
    X_trains = np.array(X_train) / 255.0
    X_trains = X_trains.reshape(X_trains.shape[0], X_trains.shape[1],
                                X_trains.shape[2], 1)

    filenamess = 'data/image/' + 'blood' + '.png'
    X_trainss = []
    image = cv2.imread(filenamess)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    X_trainss.append(image)
    X_trainss = np.array(X_trainss) / 255.0
    X_trainss = X_trainss.reshape(X_trains.shape[0], X_trains.shape[1],
                                  X_trains.shape[2], 1)

    return load_model(X_train, X_trains, X_trainss)
    # return load_model('afd', 'adsf', 'asdf')
