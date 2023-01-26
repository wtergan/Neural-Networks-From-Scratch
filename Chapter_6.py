'''Creation of a simple 2 layer neural network'''

#importation of libraries
import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data

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
        conf = np.sum(outputs_clipped * y, axis=1)
        sample_losses = -np.log(conf)
        average_loss = np.mean(sample_losses)

#get the datasets
X, y = vertical_data(samples=100, classes=3)
print('This is the first 5 samples in the dataset: \n', X[:5])
print('This is the shape of the dataset: ', X.shape)
print('This is the first 5 outputs corresponding to the first 5 samples: \n', y[:5])
print('This is the shape of the y: ', y.shape)
print('---------------------------------------------------------')

#lets plot the data
plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap='brg')
plt.show()

#lets create the model
dense1 = Dense_Layer(2, 3)
reLU = ReLU()
dense2 = Dense_Layer(3, 3)
softmax = SoftMax()
loss = CrossEntropyLoss()

