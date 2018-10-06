from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Reshape
from keras.layers import LSTM
from keras.datasets import imdb
import numpy as np

max_features = 20000
maxlen = 256  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

# print('Loading data...')
# # (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# print(len(x_train), 'train sequences')
# print(len(x_test), 'test sequences')
#
# print("train labals", len(y_train))
#
# list1 = []
# for i in range(len(y_train)):
#     if y_train[i] == 1:
#         list1.append([1, 0])
#     else:
#         list1.append([0, 1])
#
# y_train = np.array(list1)
#
# print("after process", len(y_train))
#
# print("Test labals", len(y_test))
#
# list1 = []
# for i in range(len(y_test)):
#     if y_test[i] == 1:
#         list1.append([1, 0])
#     else:
#         list1.append([0, 1])
#
# y_test = np.array(list1)
#
# print("after process", len(y_test))
#
# print('Pad sequences (samples x time)')
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)
#
# x_train = x_train.reshape((x_train.shape[0], 16, 16, 1))
# x_test = x_test.reshape((x_test.shape[0], 16, 16, 1))

Valid = [8, 134, 14, 50, 218, 6, 1543, 51, 21, 10, 50, 17, 72, 88, 50, 90, 433, 9, 167, 18317, 1029, 8, 987, 51, 840, 6, 147, 281, 7, 252, 199, 405, 3161, 732, 7, 104, 26, 1451, 4091, 17, 257, 2161, 2712, 68, 204, 732, 7, 4816, 712, 14, 3, 4951, 7, 5512, 14, 36, 26, 1200, 495, 61, 540, 1202, 2536, 3451, 7, 16495, 9, 87, 18, 3, 90, 172, 47, 14, 193, 352, 6713, 43, 11, 32, 43, 2476, 1782, 1782, 12, 143, 440, 38, 3, 63, 154, 14, 12, 80, 134, 9, 14, 49, 7, 3, 5076, 302, 33, 1842, 26, 6, 117, 3463, 2631, 12, 191, 376, 100, 1683, 139, 10, 3451, 7, 16495, 344, 2670, 3, 21, 151, 9185, 3, 541, 598, 19, 6, 645, 12337, 3681, 5573, 15348, 82, 4472, 393, 10, 3531, 6, 14559, 5003, 3489, 83, 13058, 22, 1, 7, 3062, 293, 111, 1, 33, 6, 665, 2832, 6, 3313, 124, 5484, 12318, 998, 1, 13266, 3, 116, 9, 183, 51, 13092, 17, 16495, 9, 54, 162, 17, 29, 14578, 3, 30, 2432, 46, 12, 81, 40, 3, 139, 19, 1, 32, 3, 453, 169, 40, 54, 1279, 53, 442, 1657, 31, 14, 7717, 5745, 12, 191, 30, 3, 63, 30, 1348, 12, 1275, 103, 3451, 7, 16495, 9, 6, 776, 21, 964, 722, 39, 380, 8, 1362, 87, 1285, 189, 10, 3215, 4159, 32, 63, 7304, 233, 195, 11, 114, 461, 356, 41, 753, 6, 965, 1640, 7, 1923, 106, 11, 17, 514, 17, 24, 70]
Adversarial = [8, 135, 15, 50, 218, 6, 1543, 52, 22, 11, 50, 17, 73, 88, 50, 91, 434, 9, 167, 18316, 1030, 8, 987, 52, 841, 6, 147, 281, 7, 253, 199, 406, 3161, 732, 7, 105, 26, 1451, 4091, 17, 257, 2162, 2712, 68, 205, 732, 7, 4816, 712, 15, 4, 4951, 7, 5512, 15, 36, 26, 1200, 496, 62, 540, 1203, 2536, 3452, 7, 16494, 9, 87, 18, 4, 91, 173, 47, 15, 194, 352, 6713, 44, 12, 33, 44, 2476, 1782, 1782, 13, 144, 440, 38, 4, 64, 155, 15, 13, 80, 135, 9, 15, 49, 7, 4, 5076, 302, 34, 1842, 26, 6, 117, 3463, 2631, 13, 191, 377, 101, 1683, 139, 11, 3452, 7, 16494, 345, 2670, 4, 22, 152, 9185, 4, 541, 599, 19, 6, 646, 12336, 3681, 5573, 15347, 83, 4472, 393, 11, 3532, 6, 14558, 5003, 3490, 84, 13057, 23, 2, 7, 3062, 294, 112, 2, 34, 6, 666, 2832, 6, 3314, 125, 5484, 12317, 998, 2, 13265, 4, 116, 9, 184, 52, 13091, 17, 16494, 9, 55, 163, 17, 29, 14577, 4, 31, 2433, 46, 13, 82, 40, 4, 139, 19, 2, 33, 4, 454, 169, 41, 55, 1279, 54, 442, 1658, 32, 15, 7717, 5745, 13, 191, 30, 4, 64, 31, 1348, 13, 1276, 104, 3452, 7, 16494, 9, 6, 777, 22, 964, 722, 39, 380, 8, 1363, 87, 1285, 189, 11, 3215, 4160, 33, 64, 7304, 234, 196, 12, 115, 461, 357, 42, 753, 6, 965, 1640, 7, 1923, 106, 12, 17, 515, 17, 25, 70]




def test_RNN(input_array):
    print("In TEST RNN")
    input_array = ((input_array + 0.5) * 20000).astype(int)
    print(input_array)
    model = Sequential()
    model.add(Reshape((256,), input_shape=(16, 16, 1)))
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128))#, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.load_weights("models\imdb_model.h5")
    # model.add(Dense(1, activation='sigmoid'))
    # model.add(Dense(2))
    # model.add(softmax(x, axis=-1))

    # x_train = np.array(Valid)
    # inputs = np.reshape(x_train, (1, 16, 16, 1))
    results = model.predict_classes(input_array)#, batch_size=2)
    print(results)
    return results
#result = model.predict(inputs, batch_size=None)


# model = Sequential()
# model.add(Reshape((256,), input_shape=(16, 16, 1)))
# model.add(Embedding(max_features, 128))
# model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(2, activation='softmax'))
# model.load_weights("models\imdb_model.h5")
# x_train = np.array(Adversarial)
# inputs = np.reshape(x_train, (1, 16, 16, 1))
# adv = model.predict_classes(inputs, batch_size=None)
#result = model.predict(inputs, batch_size=None)
#
# print("Valid image classification ", valid)
# print("Classification ", adv)
# print(len(set(Valid)&set(Adversarial)))
#
# # # try using different optimizers and different optimizer configs
# # model.compile(loss='binary_crossentropy',
# #               optimizer='adam',
# #               metrics=['accuracy'])
# #
# # print('Train...')
# # model.fit(x_train, y_train,
# #           batch_size=batch_size,
# #           epochs=1,
# #           # epochs=15,
# #           validation_data=(x_test, y_test))
# # score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
# # print('Test score:', score)
# # print('Test accuracy:', acc)
#
# # model.save_weights("imdb_model.h5")
# # print("Saved model to disk")