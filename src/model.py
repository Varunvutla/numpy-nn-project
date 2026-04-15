from activation import sigmoid, sigmoid_derivative

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = sigmoid(layer.forward(output))
        return output

    def train(self, X, y, epochs, lr):
        for epoch in range(epochs):
            output = self.predict(X)

            error = output - y

            for layer in reversed(self.layers):
                error = layer.backward(error * sigmoid_derivative(output), lr)

            if epoch % 1000 == 0:
                loss = ((y - output) ** 2).mean()
                print(f"Epoch {epoch}, Loss: {loss}")