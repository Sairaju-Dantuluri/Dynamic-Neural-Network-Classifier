# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


class Activations:
    def sigmoid(self, input, weights):
        z = np.matmul(input, weights.T)
        z = np.float64(z)
        return z, np.reciprocal(1 + np.exp(-z))     # 1 / (1 + e^-z)

    def tanh(self, input, weights):
        z = np.matmul(input, weights.T)
        z = np.float64(z)
        return z, (2/(1+np.exp(-2*z)))-1    # ( 2 / ( 1 + e^-z) )-1

    def ReLU(self, input, weights):
        z = np.matmul(input, weights.T)
        zf = z.copy()
        for zi in zf:
            for zj in zi:
                if zj < 0:
                    zj = 0
        return z, zf

    def LReLU(self, input, weights, leakp=0.05):
        z = np.matmul(input, weights.T)
        zf = z.copy()
        for zi in zf:
            for zj in zi:
                if zj < 0:
                    zj = leakp * zj
        return z, zf

    def dsigmoid(self, z):
        x = 1/(1+np.exp(-z))
        return (x*(1-x))

    def dtanh(self, z):
        return (1 - np.power((2/(1+np.exp(-2*z)))-1, 2))

    def dReLU(self, z):
        for zi in z:
            for zj in zi:
                if zj < 0:
                    zj = 0
                else:
                    zj = 1
        return z

    def dLReLU(self, z, p):
        for zi in z:
            for zj in zi:
                if zj < 0:
                    zj = 0
                else:
                    zj = p
        return z


class Layer:
    layerSize = None                          # layer size
    featureSize = None                        # input size that doesnt contain the bias
    activation = None                         # type of activation function to be used
    # weights of the layer (2d np array)
    weights = None
    p = 0.05                                  # leak factor for leakyReLU
    input = None
    output = None
    active = None
    z = None

    def __init__(self, layerSize, featureSize, activation='sigmoid', p=0.05):
        self.layerSize = layerSize
        self.featureSize = featureSize
        self.activation = activation
        self.input = np.ones(featureSize)
        self.p = p
        # weights of a layer dim : layer * feat+1
        self.weights = np.random.random_sample([layerSize, featureSize+1])
        self.active = Activations()

    def activfunc(self, input):
        output = np.ones([self.layerSize])
        if self.activation == 'sigmoid':
            self.z, output = self.active.sigmoid(input, self.weights)

        elif self.activation == 'tanh':
            self.z, output = self.active.tanh(input, self.weights)

        elif self.activation == 'ReLU':
            self.z, output = self.active.ReLU(input, self.weights)

        elif self.activation == 'LeakyReLU':
            self.z, output = self.active.LReLU(input, self.weights, self.p)

        else:
            print("[-] ActivFunc error : Given Activation Function is not defined")

        return output

    def dactivfunc(self):
        if self.activation == 'sigmoid':
            return self.active.dsigmoid(self.z)

        elif self.activation == 'tanh':
            return self.active.dtanh(self.z)

        elif self.activation == 'ReLU':
            return self.active.dReLU(self.z)

        elif self.activation == 'LeakyReLU':
            return self.active.dLReLU(self.z, self.p)

        else:
            print("[-] dActivFunc error : Given Activation Function is not defined")

            # input dimensions = no.of points * feat
    def out(self, input):
        # convert to atleast 2d matrix if batch size is 1.
        input = np.atleast_2d(input)
        dim = input.shape
        one = np.ones([dim[0], 1])
        input = np.hstack((one, input))          # add a '1's column
        self.input = input
        return self.activfunc(input)


class NeuralNetwork:
    inputSize = 1
    layers = []

    def __init__(self, inputlayerSize):
        # input size to build the weights of first layer nodes
        self.inputSize = inputlayerSize

    # function to add layer in the neural network, takes number of
    def addLayer(self, layerSize, activation='sigmoid', p=0.05):
        # nodes activation and leaky factor(if activ = LReLU) as parameters
        featureSize = 0
        if len(self.layers) == 0:
            featureSize = self.inputSize
        else:
            featureSize = self.layers[-1].layerSize
        self.layers.append(Layer(layerSize, featureSize, activation, p))

    # a function used by predict to convert target class to recognisable format to ANN
    def extend(self, input):
        ex = []

        for inp in input:
            inp = int(inp)
            k = self.layers[-1].layerSize
            lis = np.zeros(k)
            if k > inp and inp >= 0:
                lis[inp] = 1
            elif k <= inp:
                print("[-] Extend error : Given class >= modeloutput", inp, k)
                lis[-1] = 1
            elif inp < 0:
                print("[-] Extend error : Given class < 0")
                lis[0] = 1
            ex.append(lis)
        return np.array(ex)

    # output given by the model generally probabilities as the model is a classifier
    def predict(self, input):
        inp = input
        for layer in self.layers:
            inp = layer.out(inp)
        return inp

    # Function internally used by the ANN to train on data
    def backprop(self, delta, i):
        if i == 0:
            return None
        weidel = np.matmul(self.layers[i].weights[:, 1:].T, delta.T)
        delz = np.atleast_2d(np.average(
            self.layers[i-1].dactivfunc(), axis=0)).T

        return np.multiply(weidel, delz).T

    # Gives accuracy of the ANN on testing data.
    def accuracy(self, features, target):
        ans = self.predict(features)
        acc = 0
        for i in range(ans.shape[0]):
            if target[i] == np.argmax(ans[i]):
                acc += 1

        return acc/features.shape[0]

    def fit(self, features, target, batch_size, max_epochs, lr=0.001):
        modelError = 99999999999
        trainerr = []
        accuracy = []
        # run max_model epoch times
        for iter in range(int(max_epochs * (features.shape[0]/batch_size))):
            samp = np.hstack((features, target))
            # randomly select batch_size no. of points
            samp = samp[np.random.randint(samp.shape[0], size=batch_size), :]
            feat = samp[:, :-1]
            tar = samp[:, -1]
            pred = self.predict(feat)
            ext = self.extend(tar)

            # derivative of the log-loss w.r.t activations of final layer
            dmodelError = -np.subtract(np.multiply(ext, np.reciprocal(pred)),
                                       np.multiply((1-ext), np.reciprocal(1-pred)))

            delC = self.layers[-1].dactivfunc()
            # delta(L) for final layer
            delL = np.multiply(delC, dmodelError)
            # finding delat(l) for every layer
            for i in reversed(range(len(self.layers))):
                # delta(l-1) from delat(l)
                delL1 = self.backprop(delL, i)
                # find Cost gradient wrt each weight of layer(i-1) to layer(i)
                grad = np.matmul(delL.T, self.layers[i].input)
                self.layers[i].weights = self.layers[i].weights - \
                    (lr * grad)  # updating weights
                delL = delL1

            if iter % ((features.shape[0]/batch_size)*50) == 0:
                mpred = self.predict(features)
                mext = self.extend(target)
                modelError = np.add(np.multiply(mext, np.log(
                    mpred)), np.multiply(1-mext, np.log(1-mpred)))
                # log loss cost of the output
                modelError = -np.sum(np.sum(modelError, axis=1),
                                     axis=0)/features.shape[0]
                print("epoch :", int(
                    iter/(features.shape[0]/batch_size)), "Model error:", modelError)
                trainerr.append(modelError)

                accuracy.append(self.accuracy(features, target))

        mpred = self.predict(features)
        mext = self.extend(target)
        modelError = np.add(np.multiply(mext, np.log(mpred)),
                            np.multiply(1-mext, np.log(1-mpred)))
        modelError = -np.sum(np.sum(modelError, axis=1),
                             axis=0)/features.shape[0]
        accuracy.append(self.accuracy(features, target))
        print("Model error:", modelError)
        fig, ax = plt.subplots()
        ax.plot([50*i for i in range(len(trainerr))],
                trainerr)           # plot trainerr
        ax.set(xlabel='Epochs', ylabel='Error', title="Error vs Epochs")
        ax.grid()
        plt.show()

        fig, ax = plt.subplots()
        ax.plot([50*i for i in range(len(accuracy))],
                accuracy)           # plot accuracy
        ax.set(xlabel='Epochs', ylabel='Accuracy', title="Accuracy vs Epochs")
        ax.grid()
        plt.show()
