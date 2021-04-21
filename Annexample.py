from ann import NeuralNetwork
import pandas as pd
import numpy as np


# data preprocessing
df = pd.read_csv('dataset_NN.csv').sample(frac=1).to_numpy()
for i in [3, 5]:                                                   # normalize necessary data
    max, min = np.amax(df[:, i]), np.amin(df[:, i])
    df[:, i] = (df[:, i]-min)/(max-min)
# convert classes from [1, ..., n] to [0, ...., n-1]
df[:, -1] = df[:, -1] - 1
# split data to 70:30 ratio.
traindata, testdata = df[:int(0.7*df.shape[0])], df[int(0.7*df.shape[0]):]
# split train and test data to features and target
train_x, train_y = traindata[:, :-1], traindata[:, -1]
test_x, test_y = testdata[:, :-1], testdata[:, -1]


# ANN with 1 hidden layer

nn1 = NeuralNetwork(6)
nn1.addLayer(8, 'tanh')
nn1.addLayer(10, 'sigmoid')
nn1.fit(train_x, np.atleast_2d(train_y).T,
        batch_size=5, max_epochs=700, lr=0.01)
mpred = nn1.predict(test_x)
mext = nn1.extend(test_y)
modelError = np.add(np.multiply(mext, np.log(mpred)),
                    np.multiply(1-mext, np.log(1-mpred)))
modelError = -np.sum(np.sum(modelError, axis=1), axis=0)/test_x.shape[0]
print('Final test error', modelError)


print("\n<<<<<<<<<<<<<<<<  Accuracy  >>>>>>>>>>>>>>>>>>")
print("NN1 Test Accuracy ", nn1.accuracy(test_x, test_y))
print("NN1 Train Accuracy", nn1.accuracy(train_x, train_y), "\n")


# ANN with two hidden layers
nn2 = NeuralNetwork(6)
nn2.addLayer(7, 'sigmoid')
nn2.addLayer(9, 'sigmoid')
nn2.addLayer(10, 'sigmoid')
nn2.fit(train_x, np.atleast_2d(train_y).T,
        batch_size=5, max_epochs=700, lr=0.01)
mpred = nn2.predict(test_x)
mext = nn2.extend(test_y)
modelError = np.add(np.multiply(mext, np.log(mpred)),
                    np.multiply(1-mext, np.log(1-mpred)))
modelError = -np.sum(np.sum(modelError, axis=1), axis=0)/test_x.shape[0]
print('Final test error', modelError)


print("\n<<<<<<<<<<<<<<<<  Accuracy  >>>>>>>>>>>>>>>>>>")
print("NN2 Test Accuracy ", nn2.accuracy(test_x, test_y))
print("NN2 Train Accuracy", nn2.accuracy(train_x, train_y), "\n")
