import numpy as np
import sys


# linear combination
# v0*w0 + v1*w1 + ... + vk*wk + bias
def lin(InputVector, WeightVector, bias):
    res = 0
    for i, j in zip(InputVector, WeightVector):
        res += i * j
    return res + bias


# rectified Linear Unit
def relu(x):
    return max(0, x)


# derivative of ReLU
def relu_p(x):
    return int(x > 0)


# loss function
def err(prediction, target):
    return (prediction - target) ** 2


# learning procedure
def learn(W, b, sample, target, learning_rate):
    # dot product
    z = lin(sample, W, b)
    # activation function
    u = relu(z)
    # backpropagation
    for i in range(len(sample)):
        # dL_dw === lin'(I, w, b) with respect to 'w'
        dL_dw = sample[i]
        # dR_dL === relu'(L)
        dR_dL = relu_p(z)
        # derr_dR === d(err)/d(prediction)
        derr_dR = 2 * (u - target)
        # delta error
        derr_dw = derr_dR * dR_dL * dL_dw
        # improve weights
        W[i] -= derr_dw * learning_rate
    # dL_db === lin'(I, w, b) with respect to 'b'
    dL_db = 1
    derr_db = derr_dR * dR_dL * dL_db
    # improve bias
    b -= derr_db * learning_rate

    return W, b


# predict an output
def predict(Input, ExpectedOutput, W, b, info=True):
    z = lin(Input, W, b)
    u = relu(z)
    cost = err(u, ExpectedOutput)
    actual = ExpectedOutput
    if info:
        print("Values:    \t{}\nPrediction:\t{}".format(
            Input, u))
        print("Actual value:\t{}\nLoss:      \t{}".format(
            actual, err(u, actual)))
    return {"out": u, "cost": cost}


# dataset(x, y, z); x + y = z
dataset = [(4, 5, 9), (6, 6, 12), (2, 1, 3), (-1, 3, 2), (7, 2, 9),
           (9, 4, 13), (2, 9, 11), (5, 5, 10), (8, 1, 9), (7, 9, 16)]


def main():
    learning_rate = 0.0001
    W, b = [np.random.normal(), np.random.normal()], np.random.normal()

    if len(sys.argv) > 1:
        W = eval(sys.argv[1])
        b = eval(sys.argv[2])

    for i in range(10000):
        rand_i = np.random.randint(len(dataset))
        x = dataset[rand_i]
        W, b = learn(W, b, x[:-1], x[2], learning_rate)

    for v in dataset:
        ans = predict(v[:-1], v[2], W, b, info=False)
        print("{} + {}\t=?=\t{} ({}) - cost: {}".format(
            v[0], v[1], ans["out"], v[2], ans["cost"]))

    print("'{}' {}".format(W, b))

    # Predict 100 + 3 sum
    v = [100, 3, 103]
    ans = predict(v[:-1], v[2], W, b, info=False)
    print("{} + {}\t=?=\t{} ({}) - cost: {}".format(
        v[0], v[1], ans["out"], v[2], ans["cost"]))


''' Great weights/biases

'[0.9995483944832391, 0.9995167807995735]' 0.005130420378357128

'''


if __name__ == "__main__":
    main()
