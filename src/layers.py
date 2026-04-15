import numpy as np

class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_gradient, lr):
        weights_gradient = np.dot(self.input.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)

        self.weights -= lr * weights_gradient
        self.bias -= lr * np.sum(output_gradient, axis=0, keepdims=True)

        return input_gradient