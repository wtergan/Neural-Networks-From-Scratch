'''Backpropagation
'''

import numpy as np

class Dense_Layer:
    def __init__(self, num_inputs, num_neurons):
        #initalize the weights, biases.
        self.weights = 0.01 * np.random.randn(num_inputs, num_neurons)
        self.biases = 0.01 * np.zeros((1, num_neurons))

    def forward(self, inputs):
        #compute the matrix product of the inputs and the weights.
        self.outputs = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    #take in gradients from succeeding layer, compute gradients of parameters.
    def backward(self, dvalues):
        #gradient of inputs.
        self.dinputs = np.dot(dvalues, self.weights.T)
        #gradients of weights.
        self.dweights = np.dot(self.inputs.T, dvalues)
        #gradients of biases
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

class reLU:
    def forward(self, inputs):
        #compute the relu activation function.
        self.outputs = np.maximum(0, inputs)
        self.inputs = inputs

    def backward(self, dvalues):
        #sets gradient to 0 where input values were <= 0 during forward pass.
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class SoftMax:
    #compute the softmax activation function to map inputs to probability distribution.
    def forward(self, inputs):
        #subtracting maxima of each row from all elements of that row. for normalization.
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self):
        pass

class CCEntropyLoss:
    #compute the categorical cross entropy loss.
    def forward(self, outputs, y):
        #clip the data
        outputs_clipped = np.clip(outputs, 1e-7, 1-1e-7)
        #if true y is a one hot vector.
        if len(y.shape)==1: conf = outputs_clipped[range(len(outputs_clipped)), y]
        #if true y isnt a vector.
        elif len(y.shape)==2: conf = np.sum(outputs_clipped * y, axis=1, keepdims=True)
        #sample and average losses.
        sample_losses = -np.log(conf)
        average_loss = np.mean(sample_losses)
        return average_loss

    def backward(self):
        pass