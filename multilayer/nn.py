import numpy as np
from linear_algebra import Matrix


class Layer():
    def __init__(self, inputN, outputN):
        # weights
        self.synapses = Matrix(values=(inputN, outputN))
        # bias
        self.bias = Matrix(values=(1, outputN))
        # last evaluation of inputs, weighted sum and output
        self.inp = None
        self.weisum = None
        self.out = None

    def feedforward(self, data: Matrix):
        # save the input matrix
        self.inp = data
        # matrix multiplication plus bias
        data *= self.synapses
        data += self.bias
        # save the weighted sum evaluation
        self.weisum = data
        # activation
        data = NeuralNetwork.relu(data)
        # save the relu evaluation
        self.out = data

        return data

    # 0.5 MEAN BIAS
    def randomize(self, hasBias=True):
        # randomize synapses
        for row in range(self.synapses.rows()):
            for col in range(self.synapses.cols()):
                self.synapses.m[row][col] = np.random.normal() + 0.5
            # randomize bias
            if not hasBias:
                continue
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
        for layer in self.layers[:-1]:
            layer.randomize()
        # do not randomize output-layer bias
        self.layers[-1:][0].randomize(hasBias=False)

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
    @staticmethod
    def loss(prediction, target):
        return 0.5 * (target - prediction) ** 2

    # derivative of loss function - squared error'(p, t)
    @staticmethod
    def loss_p(prediction, target):
        return (prediction - target)

    # derivative of linear combination with respect to weights
    @staticmethod
    def dNet_dW(inp, W):
        T = ~inp
        for row in range(T.rows()):
            T.m[row] = T.m[row] * W.cols()
        return T

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
        # TODO: backpropagation algorithm
        # * check if at least 2 layers exist
        outputLayer = self.layers[-1:][0]
        inp = outputLayer.inp
        net = outputLayer.weisum
        out = outputLayer.out
        # calculate outputlayer partial derivatives
        dErr_dO = self.loss_p(out, expectedOutput)
        dO_dNet = self.relu_p(net)
        dNet_dW = self.dNet_dW(inp, outputLayer.synapses)
        # save calculation for dErr_dNet = dErr_dO * dO_dNet
        dErr_dNet = dErr_dO.HadamardProduct(dO_dNet)
        # dErr/dW = dErr/dNet * dNet/dW
        dErr_dW = dNet_dW.col_wise_mult(dErr_dNet)
        # update synapses
        outputLayer.synapses -= self.eta * dErr_dW
        # no bias update in output layer
        # hidden layers update
        # ...
        for i in range(len(self.layers) - 1, -1, -1):
            inp = self.layers[i].inp
            net = self.layers[i].weisum
            out = self.layers[i].out
            for j in range(len(self.layers[i].out)):
                # dNet/dH
                dNet_dH = self.layers[i].synapses[i][j]
            # use dErr/dNet saved from the previous calculation
            dErr_dH = dErr_dNet * dNet_dH
            # calculate partial derivatives
            dNet_dW = self.dNet_dW(inp, self.layers[i].synapses)
            # save calculation for dErr_dNet = dErr_dO * dO_dNet
            dErr_dNet = dErr_dO.HadamardProduct(dO_dNet)
            # dErr/dW = dErr/dNet * dNet/dW
            dErr_dW = dNet_dW.col_wise_mult(dErr_dNet)


# XOR Problem dataset
# dataset = {(0, 0): (0,), (0, 1): (1,), (1, 0): (1,), (1, 1): (0,)}

dataset = {(0, 0): (0, 0), (0, 1): (1, 1), (1, 0): (1, 1), (1, 1): (0, 0)}


def main():
    nn = NeuralNetwork(2, (3,), 2, learning_rate=0.01)
    nn.randomize()
    for i in range(10000):
        index = np.random.randint(len(dataset))
        inp = list(dataset.keys())[index]
        out = dataset[inp]
        nn.train(Matrix(list(inp)), Matrix(list(out)))
    for e in dataset:
        I = Matrix(list(e))
        O = Matrix(list(dataset[e]))
        pred, cost = nn.predict(I, O)
        print("{} => {}\tPred: {}\tCost: {}".format(e, dataset[e], round(pred), cost))


if __name__ == "__main__":
    main()
