## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import time

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_inception import ImageNet, InceptionModel
#from setup_rnn_keras import RNN, RNNModel
from setup_rnn_keras_imdb import RNN_Keras, RNNModel_Keras
from TestingRNN import test_RNN
import random

from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi


def show(img):
    """
    Show MNSIT digits in the console.
    """

    #print(img)
    img = ((img + 0.5) * 20000).astype(int)
    # print(img)
    seq = img.reshape(256)#(img.shape[0], 256))
    print(seq.tolist())


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    print("Targated : ", targeted)
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


if __name__ == "__main__":
    dis = []
    with tf.Session() as sess:
        #data, model = RNN(), RNNModel("models\imdb_model.h5", sess)  #MNIST(), MNISTModel("models/mnist", sess)
        data, model = RNN_Keras(), RNNModel_Keras("models\imdb_model.h5")

        attack = CarliniL2(sess, model, batch_size=1000, max_iterations=1000, confidence=0, targeted=True)

        inputs, targets = generate_data(data, samples=1000, targeted=True, start=0, inception=False)
        #show(inputs)
        timestart = time.time()
        adv = attack.attack(inputs, targets)
        timeend = time.time()

        print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

        for i in range(len(adv)):
            print("Valid:")
            show(inputs[i])
            print("Adversarial:")
            show(adv[i])

            print("Original Classification:", model.model.predict(inputs[i:i + 1]))
            print("Adversarial Classification:", model.model.predict(adv[i:i + 1]))
            distortion = np.sum((adv[i] - inputs[i]) ** 2) ** .5
            print("Total distortion:", distortion)
            dis.append(distortion)
    Original_classification = test_RNN(inputs)
    Adversarial_classification = test_RNN(adv)

    print("Original", Original_classification)
    print("Adv", Adversarial_classification)

    count = 0
    list1 = Original_classification.tolist()
    list2 = Adversarial_classification.tolist()
    for i in range(len(Original_classification)):
        if (list1[i] != list2[i]):
            count = count + 1
    print("Total Number of sample processed", len(Original_classification))
    print("L2 Accuracy :", count/len(Original_classification))
    print("Average Distortion", sum(dis)/len(dis))
