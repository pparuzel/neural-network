import numpy as np
import matplotlib.pyplot as plt


''' Activation functions '''


def relu(x):
    '''
    ReLU function
    '''
    return max(0, x)


def relu_p(x):
    '''
    Derivative of ReLU function
    '''
    return int(x > 0)


def sigmoid(x):
    '''
    Sigmoid function
    '''
    return 1 / (1 + np.exp(-x))


def sigmoid_p(x):
    '''
    Derivative of Sigmoid function
    '''
    x = sigmoid(x)
    return x * (1 - x)


def loss(prediction, target):
    '''
    Mean squared error function (loss function)
    '''
    return 0.5 * (target - prediction) ** 2


def loss_p(prediction, target):
    '''
    Derivative of mean squared error function
    '''
    return (prediction - target)


''' ReLU seems to struggle with XOR surprisingly '''
# act = np.vectorize(relu)
# act_p = np.vectorize(relu_p)


class Layer():
    def __init__(self, inputN, outputN, isOutputLayer=False):
        # amount of in's & out's
        self.inputN = inputN
        self.outputN = outputN
        # weights
        self.synapses = np.random.randn(inputN, outputN)
        # bias
        self.bias = np.random.normal() if not isOutputLayer else 0
        # last evaluation of inputs, weighted sum and output
        self.inp = None
        self.net = None
        self.out = None
        self.act = np.vectorize(sigmoid)
        self.act_p = np.vectorize(sigmoid_p)

    '''
    Feedforward algorithm
    '''

    def feedforward(self, data):
        # save the input matrix
        self.inp = data
        # matrix multiplication plus bias
        data = data.dot(self.synapses) + self.bias
        # save the weighted sum evaluation
        self.net = data
        # activation
        data = self.act(data)
        # save the activation evaluation
        self.out = data

        return data

    def __repr__(self):
        return "Layer<{}, {}>".format(self.inputN, self.outputN)


class NeuralNetwork():

    '''
    Initialize Neural Network with an amount of inputNeurons,
    hiddenNeurons, outputNeurons, and a learning rate
    '''

    def __init__(self, inputN, hiddenN, outputN, learning_rate):
        # learning rate
        self.eta = learning_rate
        # input layer
        self.layers = [Layer(inputN, hiddenN[0])]
        # hidden layers
        self.layers.extend([Layer(hiddenN[i], hiddenN[i + 1])
                            for i in range(len(hiddenN) - 1)])
        # output layer
        self.layers.append(Layer(hiddenN[-1:][0], outputN, isOutputLayer=True))

    # function propagating value through neural network
    def predict(self, inputData, expectedOutput):
        '''
        Method predicting output by feedforward algorithm
        and calculating sum of all costs
        '''
        result = self.feedforward(inputData)
        cost = loss(result, expectedOutput)
        # sum of loss function
        cost = np.sum(cost)
        return result, cost

    def feedforward(self, data):
        '''
        Method feeding data forward the network
        '''
        for layer in self.layers:
            data = layer.feedforward(data)
        return data

    def train(self, inputData, expectedOutput):
        '''
        Method training the network
        given the input and the expected output
        '''
        # return value not needed
        result = self.feedforward(inputData)
        # plot
        cost = loss(result, expectedOutput)
        cost = np.sum(cost)
        global costs
        global incr
        if incr % 1000 == 0:
            print("Error:", cost)
        incr += 1
        costs.append(cost)

        # output layer
        layer = self.layers[-1:][0]
        inp = layer.inp
        net = layer.net
        out = layer.out
        # calculate partial derivatives
        dErr_dOut = loss_p(out, expectedOutput)
        dOut_dNet = layer.act_p(net)
        # save phi
        phi = dErr_dOut * dOut_dNet
        # update weights
        layer.synapses -= self.eta * (inp.T.dot(phi))
        # hidden layer
        for li in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[li]
            inp = layer.inp
            net = layer.net
            out = layer.out
            # calculate partial derivatives
            dErr_dOut = phi.dot(self.layers[li + 1].synapses.T)
            dOut_dNet = layer.act_p(net)
            # save phi
            phi = dErr_dOut * dOut_dNet
            # update weights
            layer.synapses -= self.eta * (inp.T.dot(phi))
            # bias update
            layer.bias -= self.eta * phi


# array of costs
costs = []
# increment
incr = 0


# ELU function
def ELU(x):
    return max(x, 0.1 * (np.exp(x) - 1))


# ELU derivative
def ELU_p(x):
    return max(1, 0.1 * (np.exp(x)))


# vectorize function so that it works for matrices
act = np.vectorize(ELU)
act_p = np.vectorize(ELU_p)


def main():
    nn = NeuralNetwork(2, (2,), 1, learning_rate=0.01)
    # change activation function from sigmoid to ELU
    for layer in nn.layers:
        layer.act = act
        layer.act_p = act_p

    # XOR problem input-output dataset
    inp = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])
    out = np.array([[0],
                    [1],
                    [1],
                    [0]])
    for i in range(10000):
        nn.train(inp, out)

    pred, cost = nn.predict(inp, out)
    print("Prediction:", np.round(pred), "Loss:", cost, sep='\n')
    plt.figure(num="Costs")
    plt.title("Costs over epoch")
    plt.plot(costs)
    plt.show()


if __name__ == "__main__":
    main()
