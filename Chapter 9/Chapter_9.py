'''
Backpropagation
'''

import numpy as np

'''initalization of the inputs, weights, biases, and the dvalues, which represents
    the gradients of the next layer, this will be an array of incremental values

    remember, in order to do the matrix product of two matrices, 
    inner dimensions must be the same.
'''
inputs = np.array([[1,2,3,2.5], [2,5,-1,2], [-1.5,2.7,3.3,-0.8]])

weights = np.array([[0.2, 0.8, -0.5, 1], 
                    [0.5, -0.91, 0.26, -0.5], 
                    [-0.26,-0.27,0.17,0.87]]).T

biases = np.array([[2,3,0.5]])

dvalues = np.array([[1,1,1],[2,2,2],[3,3,3]])

print(f'inputs:\n{inputs}\n input shape: {inputs.shape}')
print(f'weights:\n{weights}\n weight shape: {weights.shape}')

'''forward pass: dot (matrix) product, add biases, reLU activation.'''
layer_outputs = np.dot(inputs, weights) + biases
relu_outputs = np.maximum(0, layer_outputs)

print(f'layer outputs before reLU activation function:\n{layer_outputs}')
print(f'layer outputs after reLU activation function:\n{relu_outputs}')


'''derivative of relu wrt the inputs values from next layer passed to current layer
    
    -make a copy of the relu output layer so this isnt affected
    -setting gradient to 0 for input values that were <=0
'''
drelu = relu_outputs.copy()
drelu[layer_outputs <= 0 ] = 0
print(f'derivative of the reLU activation function:\n{drelu}')

'''derivative of reLU wrt inputs'''
dinputs = np.dot(drelu, weights.T)

'''derivative of reLU wrt weights'''
dweights = np.dot(drelu, inputs)

'''derivative of reLU wrt. biases'''
dbiases = np.sum(drelu, axis=0, keepdims=True)

print(f'dinputs: \n{dinputs}, shape: {dinputs.shape}')
print(f'dweights: \n{dweights} shape: {dweights.shape},')
print(f'dbiases: \n{dbiases}, shape: {dbiases.shape}')

'''update the weights'''
weights = weights.T + (-0.001 * dweights)
biases += -0.001 * dbiases

print(f'current weights after back prop:\n{weights}')
print(f'current biases after back prop:\n{biases}')