# ## setup_mnist.py -- mnist data and model loading code
# ##
# ## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
# ##
# ## This program is licenced under the BSD 2-Clause licence,
# ## contained in the LICENCE file in this directory.
#
# import tensorflow as tf
# import numpy as np
# import os
# import pickle
# import gzip
# import urllib.request
# from random import randint
#
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten, Embedding, LSTM, Reshape
# from keras.layers import Conv2D, MaxPooling2D
# from keras.utils import np_utils
# from keras.models import load_model
# from keras.layers.core import K
#
# numDimensions = 300
# maxSeqLength = 256
# batchSize = 24
# lstmUnits = 64
# numClasses = 2
# iterations = 100000
#
#
# def getTrainBatch():
#     ids = np.load('idsMatrix.npy')
#
#     print(np.amax(ids))
#     max_value = np.amax(ids)
#     # added preprocessing same as MNIST as we need all values in rage of 0 to 1
#     ids2 = (ids / (max_value + 1)) - 0.5 #(data / 255) - 0.5
#     ids = ids2
#     N = ids.shape[0]
#     b = np.zeros((N, 256), dtype=float)
#     b[:ids.shape[0], :ids.shape[1]] = ids
#     ids = b
#     #print(ids.shape)
#
#     labels = []
#     arr = np.zeros([batchSize, maxSeqLength])
#     for i in range(batchSize):
#         if (i % 2 == 0):
#             num = randint(1, 11499)
#             labels.append([1, 0])
#         else:
#             num = randint(13499, 24999)
#             labels.append([0, 1])
#         arr[i] = ids[num - 1:num]
#     return arr, labels
#
#
# def getTestBatch():
#     ids = np.load('idsMatrix.npy')
#
#     print(np.amax(ids))
#     max_value = np.amax(ids)
#     # added preprocessing same as MNIST as we need all values in rage of 0 to 1
#     ids2 = (ids / (max_value + 1)) - 0.5 #(data / 255) - 0.5
#     ids = ids2
#     N = ids.shape[0]
#     b = np.zeros((N, 256), dtype=float)
#     b[:ids.shape[0], :ids.shape[1]] = ids
#     ids = b
#     #print(ids.shape)
#     labels = []
#     arr = np.zeros([batchSize, maxSeqLength])
#     for i in range(batchSize):
#         num = randint(11499, 13499)
#         if (num <= 12499):
#             labels.append([1, 0])
#         else:
#             labels.append([0, 1])
#         arr[i] = ids[num - 1:num]
#
#     #print("before reshape", arr[0].astype(int))
#     arr = arr.reshape((arr.shape[0], 16, 16, 1))
#     #print(arr[0])
#     return arr, labels
#
#
#
# class RNN_Keras:
#     def __init__(self):
#         self.test_data, self.test_labels = getTestBatch()
#
# class RNNModel_Keras:
#     def __init__(self, restore, session=None):
#         self.num_channels = 1
#         self.image_size = 16
#         self.num_labels = 2
#         max_features = 400000
#         maxlen = 256  # cut texts after this number of words (among top max_features most common words)
#
#         K.set_learning_phase(True)
#
#         model = Sequential()
#
#         # sequential.add(Reshape((1, maxlen, embedding_size)))
#         model.add(Reshape((256,), input_shape=(16, 16, 1)))
#         model.add(Embedding(max_features, 50))
#         model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
#         #model.add(LSTM(50))
#         model.add(Dense(1, activation='sigmoid'))
#         model.load_weights(restore)
#         self.model = model
#
#     def predict(self, data):
#         return self.model(data)
#
# data, model = RNN(), RNNModel("models\imdb_model.h5")
# #print(data.test_data.shape)
# #print(model.model.predict(data.test_data))