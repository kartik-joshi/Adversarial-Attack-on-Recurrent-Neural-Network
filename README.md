# Adversarial Attack on Recurrent Neural Network

## Introduction
From past few years Neural networks are gaining quite of popularity and its been used in wide range of applications ranging from autonomous cars to human face detection and language translation. Application of these algorithm to number of tasks is inversely proportional to how much easy it is easy to perform adversarial attack on Neural networks. If I can easily fool neural network, then I cannot use them in application where human lives are at stake such as autonomous vehicles.
You must be wondering what are adversarial attacks? Adversarial attack is nothing but generating an adversarial input for neural network. Such adversarial input is crafted by making intelligent perturbations to original input image. These perturbations are sufficient enough to force neural network to misclassify. Even though classification of these inputs are different, these inputs are similar to each other by some distance measure. For example, Euclidean or Manhattan distance.
I have plenty of algorithm which takes original input sequence as input to algorithm and produces adversarial example. Some of such attacks are Fast Gradient Sign, Jacobian-based Saliency Map Attack, Deep fool.
Recently, researcher form university of Berkeley introduces one of the powerful state of art adversarial example generation algorithm called C&W attack. These attacks are based on Feed forward neural networks.
Now a day’s recurrent neural network such as Long Short Term Memory (LSTM) are getting widely used in application like text summarization, voice recognition. Hence, I decided to apply these state of art C&W attack on recurrent neural network.
C&W paper has mentioned 3 attacks namely L0, L2 and L infinity. I have extended L2 attack on RNN.

## Motivation 

While reading paper “Crafting Adversarial Input Sequences for Recurrent Neural Networks”, I came across following claims that motivated us to extend current C&W attack for Recurrent neural network which are specifically developed for feed forward neural network:
1. claims that “I show that the forward derivative can be adapted to neural networks with cyclical computational graphs, using a technique named computational graph unfolding”
2. claims that “Classes of algorithms introduced previously to craft adversarial samples misclassified by feed-forward neural networks can be adapted to recurrent neural networks”


## Introduction to L2 attack
L2 attack allows smaller distortion to multiple input values to force neural network to misclassify. The L2 attack is optimization problem which tries to find adversarial example using gradient descent. L2 attack achieve this by trying to minimize following loss by searching appropriate w value.


Image 

I can divide above loss value in two part. First part of loss tries to minimize L2 distance between original and adversarial image. And second part tries to generate adversarial inputs by modifying probabilities.
In case of targeted attack, L2 tries to increase the target class probability and where as in case untagated attack L2 tries to reduce correct class probability as much as possible. Which in turn increases probability of other incorrect classes. Here in loss function tanh() tries to impose box constraint on output, which forces output not to go out of -0.5 to 0.5 value range.
The gradient descent algorithm are prone to stuck in local minima. To avoid gradient descend to get stuck in local minima, L2 attack employes multiple starting point gradient descent. In multiple point gradient descent L2 randomly picks of point closer to original image and performs gradient descent on it for some specific number of iterations.


## RNN Sentiment Analysis
As mentioned earlier to perform attack on Recurrent neural network. I took sentiment analysis problem. Sentiment analysis is widely known problem in machine learning, especially in natural language processing. In recent years I have seen good predictability in this area using LSTM recurrent neural network. Keeping that in mind I tried to build an adversarial sample for a LSTM pre trained model that does sentiment analysis. The model which I attacked was developed uses tensorflow written using keras. One of the main reason to use keras because, L2 attack implementation was written in keras. This LSTM models was build for IMDB movie review analysis. I feed the review and it predicts whether its positive or negative.

Sentiment analysis result  image 

I have IMDB movie review database which contains 25000 training and 25000 testing dataset. Divided as 50-50% for positive and negative review. Please find the model summary in above image. Sentiment analysis requires word indexing and word embedding which is taken care here using keras pre build layers. Each review is first converted into vector 256 length containing token index for given review. Here average vector size for all review is 240, so I am considering entire content for most of the review. The main reason for keeping it to 256 was because it allows us to easily integrate that to our L2 implementation. The first layer is reshape layer the input generated and feeded to our trained model in is a shape of (batch size,16,16,1). I need to reshape that and then convert it to (batch size, 256,128). Here 128 is a vector embedding, Each index is mapped with a respective vector with length of 128. Once the data reshaped I feed to a LSTM unites. Here I have total 128 LSTM units same as vector length. At the end I have dense layer that gives us value before softmax. The reason to have two nodes as the end is I need value before softmax as response to L2 attack such that I can push given adversarial sample to other class classification.

LSTM model Image 

Please find the architecture I used to build sentiment analysis model in above image. That explains how our model works and how it gives the two value from last layer for each
review.

**Result of Sentiment analysis**

| Test Accuracy | 0.86 |
| Train Accuracy | 0.98 |
| Dataset | IMDB Reviews | 
| Train samples | 25000 | 
| Test | 25000 | 
| Epoch | 8 | 


As shows in above image with 8 epoch I managed to get 98% train accuracy and 86% test accuracy.


## Modifications
The L2 attack model is divided into two components. One which is responsible for loss calculations and other is RNN on which I want to perform attack. I have done following changes to RNN or L2 to combine both.

1. Modification to input format: While training RNN I need input of batch size * number of words in sentence * word embedding size. I used batch size of 64. I set number of words in sentence as 256 because of average number of words in a sentences in our training set and word embedding size as 128.For L2 component I have input in batch size * image size * image size * channel size. Batch size is 1 for untargeted attack and it is 1000 for targeted attack. Image size is set to 16 and channel size is set to 1.
2.  Make input values in range -0.5 to 0.5: Also, for L2 component the inputs are normalized to be in the range of -0.5 to 0.5. This is in accordance with Carlini’s L2 code expectation.I did modification to the original Carlini’s code to print original and adversarial word index vector. As our project is related sentences and each word in the sentences have an index in GloVe word embedding.During this process I did investigation about word embedding for project. Investigation is like whether word embedding values are continuous or not, I found they are continuous.

3. Modified Supporting Utilities: Part from main code I have also modified supporting utilities such as functions to print original and adversarial image, functions to find accuracy and average distortion etc.

## Pre-requisites

The following steps should be sufficient to get these attacks up and running on
most Linux-based systems.

```bash
sudo apt-get install python3-pip
sudo pip3 install --upgrade pip
sudo pip3 install pillow scipy numpy tensorflow-gpu keras h5py
```

## Running attacks

```python
from robust_attacks import CarliniL2
CarliniL2(sess, model).attack(inputs, targets)
```

##And finally to test the attacks

```bash
python3 test_attack.py
```

## Attack Results 
I did testing of our code on both targeted and untargeted attack. In targeted attack I didn’t do any modifications to original input (i.e., zero perturbation). I ran this for 1000 sample and got L2 accuracy of 58.2 percent. On this 1000 sample I have average distortion of 6.51689818987.

**Targated attack result**

| No of samples | 1000 |
| Accuracy| 58.2 |
| Average distortion | 6.516 |

For untargeted attack I added little perturbation to the input and tested for sample of 10 and 20. For both the sample I got L2 accuracy of 60% for both cases. For average distortion I got values of 0.0982 and 0.0824 for 10 and 20 sample respectively.

**Untargated attack result**

| No of samples | 20 | 10 | 
| Accuracy | 0.6 | 0.6 |
| Average distortion | 0.082 | 0.0982 |

## Future Work

1. L0 and Linfinity attack: Here I have implemented L2 attack and generate adversarial input for RNN model. Next step is to generate adversarial input by using L0 and Linfinity distance. As I already integrated model with L2 code, I can integrate same model for L0 and Linfinity attack.

2.  Preserve grammar while generating adversarial input: Currently when I am generating adversarial attack using L2 attack I am not preserving grammar and semantics of given statement(review). As I am generating sentiment from word vector which neglects few words from review by keeping only words that makes a difference in sentiment. In future I can build our own word embedding and indexing which can take care of sentiment along with semantic or words.

## Conclusion
In this project I found that I can indeed do L2 attack on RNN for both targeted and untargeted attack with zero and non-zero initial perturbation respectively.

## References:
**[‘Towards Evaluating the Robustness of Neural Networks’, Nicholas Carlini David Wagner](https://arxiv.org/abs/1608.04644)
**[‘Crafting Adversarial Input Sequences for Recurrent Neural Networks’, Nicolas Papernot and Patrick McDaniel](https://arxiv.org/abs/1604.08275)
**[Sentiment analysis with LSTM using TensorFlow](https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow)
