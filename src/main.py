from layers import Dense
from model import NeuralNetwork
from generate_data import get_xor_data

X, y = get_xor_data()

network = NeuralNetwork([
    Dense(2, 4),
    Dense(4, 1)
])

network.train(X, y, epochs=10000, lr=0.1)

print("Final Output:")
print(network.predict(X))