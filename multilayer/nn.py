import numpy as np
from linear_algebra import Matrix


class Layer():
    def __init__(self, inputN, outputN):
        # weights
        self.synapses = Matrix(values=(inputN, outputN))
        # bias
        self.bias = Matrix(values=(1, outputN))
        # last evaluation of inputs weighted sum and output
        self.inp = None
        self.weisum = None
        self.out = None

    def feedforward(self, data: Matrix):
        # save the input matrix
        self.inp = data
        # matrix multiplication plus bias
        data = data * self.synapses + self.bias
        # save the weighted sum evaluation
        self.weisum = data
        # activation
        data = NeuralNetwork.relu(data)
        # save the relu evaluation
        self.out = data

        return data

    # 0.5 MEAN BIAS
    def randomize(self):
        # randomize synapses
        for row in range(self.synapses.rows()):
            for col in range(self.synapses.cols()):
                self.synapses.m[row][col] = np.random.normal() + 0.5
            # randomize bias
            for rowb in range(self.bias.rows()):
                for colb in range(self.bias.cols()):
                    self.bias.m[rowb][0] = np.random.normal() + 0.5

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
        # do not randomize output-layer bias
        for layer in self.layers:
            layer.randomize()

    # activation function - ReLU(x)
    @staticmethod
    def relu(x):
        if isinstance(x, Matrix):
            for row in range(x.rows()):
                for col in range(x.cols()):
                    x.m[row][col] = max(0, x.m[row][col])
            return x
        else:
            return max(0, x)

    # derivative of activation function - ReLU'(x)
    @staticmethod
    def relu_p(x):
        if isinstance(x, Matrix):
            for row in range(x.rows()):
                for col in range(x.cols()):
                    x.m[row][col] = 1 if x.m[row][col] > 0 else 0
            return x
        else:
            return 1 if x > 0 else 0

    # loss function - squared error(p, t)
    def loss(self, prediction, target):
        return (target - prediction) ** 2

    # derivative of loss function - squared error'(p, t)
    def loss_p(self, prediction, target):
        return 2 * (prediction - target)

    # function propagating value through neural network
    def predict(self, inputData, expectedOutput):
        for layer in self.layers:
            inputData = layer.feedforward(inputData)
        cost = self.loss(inputData, expectedOutput)
        # sum of loss function TODO: is it correct?
        cost = sum(cost.m[0])
        return inputData, cost

    # feedforward and apply backpropagation
    def train(self, inputData, expectedOutput):
        prediction, cost = self.predict(inputData, expectedOutput)
        print("Pred: {}\nCost: {}".format(prediction, cost))
        # TODO: backpropagation algorithm
        # output layer update
        # check if at least 2 layers exist
        outputLayer = self.layers[-2:][1]
        hiddenLayer = self.layers[-2:][0]
        inp = outputLayer.inp
        lin = outputLayer.weisum
        out = outputLayer.out
        dErr_dO = self.loss_p(out, expectedOutput)
        dO_dZ = self.relu_p(lin)
        # TODO: macierz wag wyjedynkuj
        #       przemnoz przez input
        dZ_dW = self.dLin_dW(inp, outputLayer.synapses)
        dErr_dW = dErr_dO * dO_dZ * dZ_dW
        # ...
        # hidden layers update
        # ...
        for i in range(len(self.layers) - 1, -1, -1):
            lin = self.layers[i].weisum
            out = self.layers[i].out
            dErr_dR = self.loss_p(out, expectedOutput)
            dR_dZ = self.relu_p(lin)
            # dZ_dW = ktoras value przy wadze W
            dErr_dW = dErr_dR * dR_dZ * dZ_dW


def main():
    mlp = NeuralNetwork(2, (3,), 2, learning_rate=0.0001)
    mlp.randomize()
    mlp.train(Matrix([1, 2]), Matrix([3, 4]))


if __name__ == "__main__":
    main()
