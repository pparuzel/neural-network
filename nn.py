''' Simple Neural Network '''

import numpy as np
import sys


# Fifa players (AGE, OVERALL, STAMINA)
dataset = [(32, 94, 92), (30, 93, 73), (25, 92, 78), (30, 92, 89), (31, 92, 44),
           (28, 91, 79), (26, 90, 40), (26, 90, 79), (27, 90, 77), (29, 90, 72)]


#   age, overall, stamina
sergio_ramos = (31, 90, 84)


def progressbar(i):
    print(" [{:<10}] {: 3}%".format('=' * int(i / 10),
                                    i), flush=True, end="\r")


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
        print("Data (Age, Overall): {}\nPrediction (Stamina): {}".format(
            Input, u))
        print("Actual value: {}\nLoss: {}".format(actual, err(u, actual)))
    return {"out": u, "cost": cost}


# main function
def main():
    W, b = [np.random.randn(), np.random.randn()], np.random.randn()
    learning_rate = 0.00001
    reps = 100000

    if len(sys.argv) > 1:
        W = eval(sys.argv[1])
        b = eval(sys.argv[2])

    print("Weights, Bias: '{}' {} ".format(W, b))
    PERCENT = 0
    print()
    ''' NN Begin '''
    for i in range(reps):
        rand_i = np.random.randint(len(dataset))
        rp = dataset[rand_i]
        Inp = rp[:-1]
        # dot product
        z = lin(Inp, W, b)
        # activation function
        u = relu(z)
        if u == 0:
            print("NULL")
            return
        # loss function (doesn't need to be calculated)
        # cost = err(u, rp[2])
        if i % (reps / 100) == 0:
            progressbar(PERCENT)
            PERCENT += 1
        # backpropagation
        for i in range(len(Inp)):
            # dL_dw === lin'(I, w, b) with respect to 'w'
            dL_dw = Inp[i]
            # dR_dL === relu'(L)
            dR_dL = relu_p(z)
            # derr_dR === d(err)/d(prediction)
            derr_dR = 2 * (u - rp[2])
            # delta error
            derr_dw = derr_dR * dR_dL * dL_dw
            # improve weights
            W[i] -= derr_dw * learning_rate
        # dL_db === lin'(I, w, b) with respect to 'b'
        dL_db = 1
        derr_db = derr_dR * dR_dL * dL_db
        # improve bias
        b -= derr_db * learning_rate
    ''' NN End '''
    for _d in dataset:
        predict(_d[:-1], _d[2], W, b)
        print()
    print("Better Weights, Bias: '{}' {} ".format(W, b))
    # # learn only one data sample
    # for i in range(100000):
    #     W, b = learn(W, b, dataset[0][:-1], dataset[0][2], learning_rate)

    # try to predict
    print("\n\tSergio Ramos")
    predict(sergio_ramos[:-1], sergio_ramos[2], W, b)


''' Actually some good weights '''
#
# '[0.19556497236564455, 0.5498677344459515]' 0.44030114456510305
# '[0.19556497236564455, 0.5498677344459515]' 16.365692369684496
# '[0.597030460468529, 0.49790255547512696]' 15.16553737709312
# '[0.7367558056631766, 0.45169329115720835]' 13.96682537688979 <= GREAT
# '[0.7070670893361005, 0.47110790104844474]' 13.846113688647153
#
''' '''

if __name__ == "__main__":
    main()
