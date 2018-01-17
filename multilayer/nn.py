import numpy as np
from linear_algebra import Matrix


class Layer():
    def __init__(self, inputN, outputN, bias=0):
        # weights
        self.synapses = Matrix(values=(inputN, outputN))
        # bias
        self.bias = Matrix(values=(outputN, 1))

    def feedforward(self, data: Matrix):
        # dot product
        data = Matrix.dot(data, self.synapses, self.bias)
        # activation
        return NeuralNetwork.relu(data)

    def __repr__(self):
        return "Layer<{!r}>".format(self.synapses)


class NeuralNetwork():
    # Initialize Neural Network with an amount of inputNeurons,
    # hiddenNeurons, outputNeurons, and a learning rate
    def __init__(self, inputN, hiddenN, outputN, learning_rate):
        # learning rate
        self.eta = learning_rate
        # input layer
        self.layers = [Layer(inputN, hiddenN[0])]
        # hidden layers
        self.layers.extend([Layer(hiddenN[i], hiddenN[i + 1])
                            for i in range(len(hiddenN) - 1)])
        # output layer
        self.layers.append(Layer(hiddenN[-1:][0], outputN))

    def randomize(self):
        for layer in self.layers:
            # randomize synapses
            for row in range(layer.synapses.rows()):
                for col in range(layer.synapses.cols()):
                    layer.synapses.m[row][col] = np.random.normal()
            # randomize bias
            for row in range(bias.rows()):
                bias.m[row][0] = np.random.normal()

    @staticmethod
    def relu(x):
        return max(0, x)

    def loss(self, prediction, target):
        return (prediction - target) ** 2

    def predict(self, inputData, expectedOutput):
        data = inputData
        for layer in self.layers:
            data = layer.feedforward(data)
        cost = self.loss(data, expectedOutput)
        return data, cost

    def train(self, inputData, expectedOutput):
        prediction, cost = self.predict(inputData, expectedOutput)
        print("Pred: {}\tCost: {}".format(prediction, cost))
        # backpropagation algorithm


def main():
    mlp = NeuralNetwork(2, (2,), 1, learning_rate=0.0001)
    mlp.randomize()
    mlp.train(Matrix([1, 2]), Matrix([3]))
    # mlp.predict(Matrix([1, 2])


if __name__ == "__main__":
    main()
