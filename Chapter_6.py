'''Creation of a Neural Network'''

#importation of libraries
import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data

#creation of the classes need for the neural network
class Dense_Layer:
    def __init__(self, num_features, num_neurons):
        self.weights = 0.01 * np.random.randn(num_features, num_neurons)
        self.biases = np.zeros((1, num_neurons))

    def forward(self, samples):
        self.outputs = np.dot(samples, self.weights) + self.biases

class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class SoftMax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.outputs = exp_values / np.sum(exp_values)

class CrossEntropyLoss:
    def forward(self, outputs, y):
        outputs_clip = np.clip(outputs, 1e-7, 1-1e-7)
        conf = np.sum(outputs_clip * y, axis=1)
        sample_losses = -np.log(conf)
        average_loss = np.mean(sample_losses)
        return average_loss

#get the data
X, y = vertical_data(samples=100, classes=3)
print('this is first 5 samples in the dataset: \n', X[:5], 'shape: ', X.shape)
print('this is the first 5 values in the y: \n', y[:5], 'shape: ', y.shape)
     