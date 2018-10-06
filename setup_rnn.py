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

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model

# ids = np.load('idsMatrix.npy')
# import numpy as np
# N = ids.shape[0]
# b = np.zeros((N, 256),dtype=int)
# b[:ids.shape[0],:ids.shape[1]] = ids
# ids=b
# print(ids.shape)


# def extract_data(filename, num_images):
#     with gzip.open(filename) as bytestream:
#         bytestream.read(16)
#         buf = bytestream.read(num_images * 28 * 28)
#         data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
#         data = (data / 255) - 0.5
#         data = data.reshape(num_images, 28, 28, 1)
#         return data
#
#
# def extract_labels(filename, num_images):
#     with gzip.open(filename) as bytestream:
#         bytestream.read(8)
#         buf = bytestream.read(1 * num_images)
#         labels = np.frombuffer(buf, dtype=np.uint8)
#     return (np.arange(10) == labels[:, None]).astype(np.float32)


from random import randint
import numpy as np
import tensorflow as tf


#ids = np.load('idsMatrix.npy')
#print(ids[0])

numDimensions = 300
maxSeqLength = 256
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000


def getTrainBatch():
    ids = np.load('idsMatrix.npy')
    N = ids.shape[0]
    b = np.zeros((N, 256), dtype=float)
    b[:ids.shape[0], :ids.shape[1]] = ids
    ids = b
    #print(ids.shape)
    labels = []
    print(ids.shape)
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(1, 11499)
            labels.append([1, 0])
        else:
            num = randint(13499, 24999)
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels


def getTestBatch():
    ids = np.load('idsMatrix.npy')
    N = ids.shape[0]
    b = np.zeros((N, 256), dtype=float)
    b[:ids.shape[0], :ids.shape[1]] = ids
    ids = b
    #print(ids.shape)
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(11499, 13499)
        if (num <= 12499):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]

    #print("before reshape", arr[0].astype(int))
    arr = arr.reshape((arr.shape[0], 16, 16))
    return arr, labels


class RNN_tf:
    def __init__(self):
        # if not os.path.exists("data"):
        #     os.mkdir("data")
        #     files = ["train-images-idx3-ubyte.gz",
        #              "t10k-images-idx3-ubyte.gz",
        #              "train-labels-idx1-ubyte.gz",
        #              "t10k-labels-idx1-ubyte.gz"]
        #     for name in files:
        #         urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/" + name)
        #
        # train_data = extract_data("data/train-images-idx3-ubyte.gz", 60000)
        # train_labels = extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
        # self.test_data = extract_data("data/t10k-images-idx3-ubyte.gz", 10000)
        # self.test_labels = extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)
        #
        # VALIDATION_SIZE = 5000
        #
        # self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        # self.validation_labels = train_labels[:VALIDATION_SIZE]
        # self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        # self.train_labels = train_labels[VALIDATION_SIZE:]

        self.test_data, self.test_labels = getTestBatch()

class RNNModel_tf:
    def __init__(self, restore=None, session=None):
        self.num_channels = 1
        self.image_size = 16
        self.num_labels = 2
        maxSeqLength_for_RNN = 250

        wordsList = np.load('wordsList.npy').tolist()
        wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
        self.wordVectors = np.load('wordVectors.npy')
        #tf.reset_default_graph()
        #labels = tf.placeholder(tf.float32, [batchSize, numClasses])
        #input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength_for_RNN])
        #data = tf.Variable(tf.zeros([batchSize, maxSeqLength_for_RNN, numDimensions]), dtype=tf.float32)
        data = tf.nn.embedding_lookup(self.wordVectors, self.input_data)
        lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
        value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
        weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
        bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        prediction = (tf.matmul(last, weight) + bias)

        #with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('models'))

        self.model = prediction

    def predict(self, input, sess):
        return self.model(input)
        #return self.model(data)
        #print(input)
        #print(input.shape)
        # self.num_channels = 1
        # self.image_size = 16
        # self.num_labels = 2
        # maxSeqLength_for_RNN = 250
        #
        # wordsList = np.load('wordsList.npy').tolist()
        # wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
        # wordVectors = np.load('wordVectors.npy')
        # #tf.reset_default_graph()
        # labels = tf.placeholder(tf.float32, [batchSize, numClasses])
        # input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength_for_RNN])
        # data = tf.Variable(tf.zeros([batchSize, maxSeqLength_for_RNN, numDimensions]), dtype=tf.float32)
        # data = tf.nn.embedding_lookup(wordVectors, input_data)
        # lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
        # lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
        # value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
        # weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
        # bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
        # value = tf.transpose(value, [1, 0, 2])
        # last = tf.gather(value, int(value.get_shape()[0]) - 1)
        # prediction = (tf.matmul(last, weight) + bias)
        #
        # #with tf.Session() as sess:
        # #sess.run(tf.global_variables_initializer())
        # saver = tf.train.Saver()
        # saver.restore(sess, tf.train.latest_checkpoint('models'))
        #
        # # reshape to feed to RNN
        # t = np.reshape(input, (input.shape[0], -1))
        # input = t[:, 0:250]
        #
        # res = sess.run(prediction, {input_data: input})
        # return res

rnn, rnnModel = RNN_tf(), RNNModel_tf()
with tf.Session() as sess:
    print(rnnModel.predict(rnn.test_data, sess))