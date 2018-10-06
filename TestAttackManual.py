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

Valid = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 89, 27, 1, 9289, 17, 199, 131, 4, 4191, 16, 1339, 23, 8, 759, 3, 1385, 7, 3, 21, 1368, 11415, 16, 5149, 17, 1634, 7, 1, 1368, 9, 3, 1356, 8, 13, 990, 12, 877, 38, 19, 27, 239, 12, 100, 234, 60, 483, 11960, 3, 7, 3, 20, 131, 1101, 71, 8, 13, 251, 27, 1146, 7, 308, 16, 735, 1516, 17, 29, 143, 28, 77, 2305, 18, 11]

Adversarial = [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]

model = Sequential()
model.add(Reshape((256,), input_shape=(16, 16, 1)))
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.load_weights("models\imdb_model.h5")
# model.add(Dense(1, activation='sigmoid'))
# model.add(Dense(2))
# model.add(softmax(x, axis=-1))

x_train = np.array(Valid)
inputs = np.reshape(x_train, (1, 16, 16, 1))
valid = model.predict_classes(inputs, batch_size=None)
#result = model.predict(inputs, batch_size=None)


x_train = np.array(Adversarial)
inputs = np.reshape(x_train, (1, 16, 16, 1))
adv = model.predict_classes(inputs, batch_size=None)
#result = model.predict(inputs, batch_size=None)

print("Valid image classification ", valid)
print("Adversarial image classification ", adv)
print("same words", len(set(Valid)&set(Adversarial)))

# # try using different optimizers and different optimizer configs
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# print('Train...')
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=1,
#           # epochs=15,
#           validation_data=(x_test, y_test))
# score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
# print('Test score:', score)
# print('Test accuracy:', acc)

# model.save_weights("imdb_model.h5")
# print("Saved model to disk")