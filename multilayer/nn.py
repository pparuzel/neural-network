import numpy as np
from linear_algebra import Matrix


class Layer():
    def __init__(self, inputN, outputN):
        # weights
        self.synapses = Matrix(values=(inputN, outputN))
        # bias
        self.bias = Matrix(values=(1, outputN))

    def feedforward(self, data: Matrix):
        # matrix multiplication plus bias
        data = data * self.synapses + self.bias
        # activation
        return NeuralNetwork.relu(data)

    # 0.5 STARTING BIAS
    def randomize(self):
        # randomize synapses
        for row in range(self.synapses.rows()):
            for col in range(self.synapses.cols()):
                self.synapses.m[row][col] = np.random.normal() + 0.5
            # randomize bias
            for row in range(self.bias.rows()):
                self.bias.m[row][0] = np.random.normal() + 0.5

    def __repr__(self):
        return "Layer<{!r}>".format(self.synapses)


class NeuralNetwork():
    # initialize Neural Network with an amount of inputNeurons,
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

    # randomize synapses and bias
    def randomize(self):
        for layer in self.layers:
            layer.randomize()

    # activation function - ReLU
    @staticmethod
    def relu(x):
        if isinstance(x, Matrix):
            for row in range(x.rows()):
                for col in range(x.cols()):
                    x.m[row][col] = max(0, x.m[row][col])
            return x
        else:
            return max(0, x)

    # loss function - squared error
    def loss(self, prediction, target):
        return (prediction - target) ** 2

    # function propagating value through neural network
    def predict(self, inputData, expectedOutput):
        for layer in self.layers:
            inputData = layer.feedforward(inputData)
        cost = self.loss(inputData, expectedOutput)
        return inputData, cost

    # feedforward and apply backpropagation
    def train(self, inputData, expectedOutput):
        prediction, cost = self.predict(inputData, expectedOutput)
        print("Pred: {}\nCost: {}".format(prediction, cost))
        # TODO: backpropagation algorithm
        # ...


def main():
    mlp = NeuralNetwork(2, (3,), 1, learning_rate=0.0001)
    mlp.randomize()
    mlp.train(Matrix([1, 2]), Matrix([3]))


if __name__ == "__main__":
    main()
