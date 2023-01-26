'''Creation of a simple 2 layer neural network'''

#importation of libraries
import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

#creation of the Dense Layer Class
class Dense_Layer:
    def __init__(self, num_features, num_neurons):
        self.weights = np.random.randn(num_features, num_neurons)
        self.biases = np.zeros((1, num_neurons))

    def forward(self, samples):
        self.outputs = np.dot(samples, self.weights) + self.biases

#creation of the ReLU Activation Function Class
class ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)

#creation of the SoftMax Activation Function Class
class SoftMax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.outputs = exp_values / np.sum(exp_values) 

#creation of the Categorical Cross Entropy Loss Function Class
class CrossEntropyLoss:
    def forward(self, outputs, y):
        outputs_clipped = np.clip(outputs, 1e-7, 1-1e-7)
        if len(y.shape) == 1:
            correct_confidences = outputs_clipped[range(len(outputs)), y]
        elif len(y.shape) == 2:
            correct_confidences =  np.sum(outputs_clipped * y, axis=1)

        sample_losses = -np.log(correct_confidences)
        average_loss = np.mean(sample_losses)
        return average_loss

def train(dense1, reLU, dense2, softmax, loss):
    '''Training loop:
    if the current set of weights and biases result in a better performance
    than the previous best set, save those as the new best set.
    
    loop continues to run for n iterations, until it finds the best set
    of results int he lowest loss and the highest accuracy
    
    .copy() creates a copy of the weights and biases of the dense layers
    after a new set of weights and biases that result in a lower loss
    is found. This way, even if the original weights and biases are modified
    later on, the copy of the best set is still saved and can be used 
    to revert back to the best set if a new set doesn't result in a lower loss'''

    #helper variables to track the best lost, and its associated weights, biases
    lowest_loss = 999999 #some initial value
    best_dense1_weights = dense1.weights.copy()
    best_dense1_biases = dense1.biases.copy()
    best_dense2_weights = dense2.weights.copy()
    best_dense2_biases = dense2.biases.copy()

    for iteration in range(100000):
        #update the weights with some small values
        dense1.weights += 0.05 * np.random.randn(2, 3)
        dense1.biases += 0.05 * np.random.randn(1, 3)
        dense2.weights += 0.05 * np.random.randn(3, 3)
        dense2.biases += 0.05 * np.random.randn(1, 3)

        #perform the forward passs of the training data through the layers
        dense1.forward(X)
        reLU.forward(dense1.outputs)
        dense2.forward(reLU.outputs)
        softmax.forward(dense2.outputs)

        #perform a forward pass through the loss function, returns the loss
        loss_output = loss.forward(softmax.outputs, y)

        '''calculate the accuracy from the softmax outputs and the targets
        
            np.argmax() returns the indices of the max value along each rows (sample)
            predictions == y compates the predicted class for each sample
            in the softmax outouts. returns an array of True/False Values
            np.mean() average: # of correct predictions / the total # predictions'''
        predictions = np.argmax(softmax.outputs, axis=1)
        accuracy = np.mean(predictions == y)

        #if loss is smaller, print and save weights and biases aside
        if loss_output < lowest_loss:
            print('New set of weights found, iteration: ', iteration, 'loss: ', loss_output, 'acc: ', accuracy)
            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases = dense1.biases.copy()
            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases = dense2.biases.copy()
        #else, revert weights and losses seince the new set doesn't result in a lower loss
        else:
            dense1.weights = best_dense1_weights.copy()
            dense1.biases = best_dense1_biases.copy()
            dense2.weights = best_dense2_weights.copy()
            dense2.biases = best_dense2_biases.copy()

def model_creation():
    #lets create the model
    dense1 = Dense_Layer(2, 3)
    reLU = ReLU()
    dense2 = Dense_Layer(3, 3)
    softmax = SoftMax()
    loss = CrossEntropyLoss()
    return dense1, reLU, dense2, softmax, loss

#get the datasets
X, y = spiral_data(samples=100, classes=3)
print('This is the first 5 samples in the dataset: \n', X[:5])
print('This is the shape of the dataset: ', X.shape)
print('This is the first 5 outputs corresponding to the first 5 samples: \n', y[:5])
print('This is the shape of the y: ', y.shape)
print('---------------------------------------------------------')

#lets plot the data
plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap='brg')
plt.show()

#train the model:
dense1, reLU, dense2, softmax, loss = model_creation()
train(dense1, reLU, dense2, softmax, loss)
