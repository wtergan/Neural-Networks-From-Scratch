''''Backpropagation'''

import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data

nnfs.init()

#creation of the Dense Layer class, used to make a layer of a neuron.
class Dense_Layer:
    #initalization of weights and biases.
    def __init__(self, num_inputs, num_neurons):
        self.weights = 0.01 * np.random.randn(num_inputs, num_neurons)
        self.biases = 0.01 * np.zeros((1,num_neurons))

    #perform the matrix product of the inputs and the weights.
    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
    
    #take in gradients of next layer, get gradients of parameters.
    def backward(self, dvalues):
        self.dinputs = np.dot(dvalues, self.weights.T)
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.biases = np.sum(dvalues, axis=0, keepdims=True)

#creation of the reLU activation function.
class reLU:
    #perform the reLU on the outputs of the dense layer.
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)
        self.inputs = inputs

    #take in gradients of next layer, take the derivative of reLU wrt. inputs.
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        #if input value is negative, set the gradient that corresponds to that value as 0.
        self.dinputs[self.inputs <= 0] = 0

#creation of the SoftMax activation function.
class SoftMax:
    #compute the softmax to get the probability distribution of the outputs.
    def forward(self, inputs):
        #normalize the exp values by subtracting each max value from each row with the row.
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues):
        pass

#creation of the Categorical Cross Entropy Loss function.
class CCrossEntropyLoss:
    #compute the loss of the outputs from the softmax and the true label class.
    def forward(self, outputs, y):
        #clip the data so that no outputs are exactly 0 or 1.
        outputs_clipped = np.clip(outputs, 1e-7, 1-1e-7)
        #compute the confidence scores vector depending on the shape y.
        if len(y.shape)==1: conf = outputs_clipped[range(len(outputs)), y]
        elif len(y.shape)==2: conf = np.sum(outputs_clipped*y, axis=1)
        #compute the sample and average losses.
        sample_losses = -np.log(conf)
        average_loss = np.mean(sample_losses)

    #compute the derivative of the loss function wrt its inputs (predicted class prob.)
    def backward(self, dvalues, y):
        #get the number of samples.
        samples = len(dvalues)
        #number of labels in every sample. we will use the first samples to count them.
        labels = len(dvalues[0])
        #if labels are sparse, turn them into a one-hot vector.
        if len(y.shape)==1: y = np.eye(labels)[y]
        #compute the gradient. then normalize.
        self.dinputs = -y/dvalues
        self.dinputs = self.inputs / samples