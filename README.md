# Dynamic-Neural-Network-Classifier

## Model details
The code uses log-loss function as cost function and Mini batch GD as training function.

## Running the code

Import the object NeuralNetwork from ann.py
NeuralNetwork object takes no.of features of input for initialisation.

NeuralNetwork has:
### addLayer()
This method is used to add hidden layers and output layer
The method takes an integer specifing layersize , an optional activation function : sigmoid/tanh/ReLU/LeakyReLU (default : sigmoid) 
and an optional p value (parameter for leakyReLU) (default : 0.05)

### fit()
This method is used to train the data
This takes the training features and targets along with batch size, max_epochs and learning rate as parameters

### predict()
This method is used to get the output of the Neural network.
This takes the features as input.

### accuracy()
This method gives the accuracy of classification of the model when given features and respective expected classes.

## PS: every method can take multiple data points at the sametime as they take numpy.arrays as input
example: [[1,3,4],
          [2,5,6]] is a numpy array with 2 datapoints and having 3 features each.
          
## Annexample.py is a sample file to build and train NeuralNetworks
