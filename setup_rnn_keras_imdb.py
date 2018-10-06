## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import urllib.request
from random import randint

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding, LSTM, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras.layers.core import K


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import numpy as np

from keras.datasets import imdb
import numpy as np

max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')


numDimensions = 300
maxSeqLength = 256
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000

def getData():
    max_features = 20000
    maxlen = 256  # cut texts after this number of words (among top max_features most common words)
    batch_size = 32

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print("train labals", len(y_train))

    list1 = []
    for i in range(len(y_train)):
        if y_train[i] == 1:
            list1.append([1, 0])
        else:
            list1.append([0, 1])

    y_train = np.array(list1)

    print("after process", len(y_train))

    print("Test labals", len(y_test))

    list1 = []
    for i in range(len(y_test)):
        if y_test[i] == 1:
            list1.append([1, 0])
        else:
            list1.append([0, 1])

    y_test = np.array(list1)

    print("after process", len(y_test))

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def getTrainBatch():
    x_train, y_train, x_test, y_test = getData()
    print("Max value", np.max(x_train))
    x_train = (x_train / 20000) - 0.5
    x_train = x_train.reshape((x_train.shape[0], 16, 16, 1))
    return x_train, y_train


def getTestBatch():
    x_train, y_train, x_test, y_test = getData()
    print("Max value", np.max(x_train))
    x_test = (x_test / 20000) - 0.5
    x_test = x_test.reshape((x_test.shape[0], 16, 16, 1))
    return x_test, y_test



class RNN_Keras:
    def __init__(self):
        self.test_data, self.test_labels = getTestBatch()

class RNNModel_Keras:
    def __init__(self, restore, session=None):
        self.num_channels = 128
        self.image_size = 16
        self.num_labels = 2
        max_features = 20000
        maxlen = 256  # cut texts after this number of words (among top max_features most common words)

        K.set_learning_phase(True)

        print('Build model...')
        model = Sequential()
        model.add(Reshape((256,), input_shape=(16, 16, 1)))
        model.add(Embedding(max_features, 128))
        model.add(LSTM(128))
        model.add(Dense(2))#, activation='softmax'))


        model.load_weights(restore)
        self.model = model

    def predict(self, data):
        return self.model(data)

#data = RNN_Keras()
#getTrainBatch()
# data, model = RNN_Keras(), RNNModel_Keras("models\imdb_model.h5")
# print(data.test_data.shape)
# print(model.predict(data.test_data))