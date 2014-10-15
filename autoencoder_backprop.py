from numpy import *
import scipy.special
import math


def sigmoid_deriv(x):
    ex = exp(-x)
    y = ex / (1 + ex)**2
    return y


def KLsum(sequence, desiredRho):
    mean = sequence.mean(1)  # the mean activation of every colomn
    s = 0
    for num in mean:
        s += num*math.log(num/desiredRho) + \
            (1-num)*math.log((1-num)/(1-desiredRho))
    return s


def backpropagation(visibleSize, hiddenSize, numData,
                    W1, W2, b1, b2, trainingSet, weightDecayLambda=0,
                    sparsity_param=0.5, sparsity_weight_beta=3):
    # GRAD INITIALIZATION
    W1grad = zeros((hiddenSize, visibleSize))
    W2grad = zeros((visibleSize, hiddenSize))

    # FORWARD PASS
    hidZ = dot(W1, trainingSet)+repeat(b1, numData, 1)
    hidA = scipy.special.expit(hidZ)
    # this is the sigmoid function
    outZ = dot(W2, hidA)+repeat(b2, numData, 1)
    out = scipy.special.expit(outZ)

    # COST CALCULATION
    errors = [linalg.norm(out[:, i]-trainingSet[:, i]) for i in range(numData)]
    cost = 0.5*sum(errors)
    # add the weight decay term
    cost += (weightDecayLambda/2.0)*(sum(sum(W1**2))+sum(sum(W2**2)))
    # add the sparsity penalty term
    cost += KLsum(hidA, sparsity_param)*sparsity_weight_beta

    # DELTA CALCULATION
    # delta for output layer
    deltaOutput = (-(trainingSet-out)) * sigmoid_deriv(outZ)
    # visibleSize by numData

    rho = mean(hidA, 1)
    sparsity_grad_term = sparsity_weight_beta * \
        (-rho/sparsity_param + (1-rho)/(1-sparsity_param))
    sparsity_grad_term = sparsity_grad_term.reshape((hiddenSize, 1))
    sparsity_grad_term = repeat(sparsity_grad_term, numData, 1)
    deltaHidden = (dot(W2.transpose(), deltaOutput) + sparsity_grad_term) \
        * sigmoid_deriv(hidZ)
    # hiddenSize by numData
    # without sparsity constraint
    # deltaHidden = dot(W2.transpose(), deltaOutput) * sigmoid_deriv(hidZ)

    # the partial W partial J(W, b, x, y) x is the ith data
    # activation[previous_layer] times deltaOftheLayer
    # sum over all that

    for i in range(numData):
        W1grad += outer(deltaHidden[:, i], trainingSet[:, i])
        W2grad += outer(deltaOutput[:, i], hidA[:, i])

    W1grad /= numData
    W2grad /= numData
    b1grad = deltaHidden.sum(axis=1)/numData
    b2grad = deltaOutput.sum(axis=1)/numData

    return W1grad, W2grad, b1grad, b2grad, cost


def train(trainingSet, hiddenSize, learningRate,
          maxIteration=250000, weightDecayLambda=0):
    visibleSize = trainingSet.shape[0]
    numData = trainingSet.shape[1]
    # print(("numData", numData))

    # INITIALIZATION
    W1 = zeros((hiddenSize, visibleSize))
    # from input to the hidden layer, so hiddenSize by visibleSize
    b1 = zeros((hiddenSize, 1))
    W2 = zeros((visibleSize, hiddenSize))
    # from hidden layer to output layer
    b2 = zeros((visibleSize, 1))
    it = 0

    while True:
        W1grad, W2grad, b1grad, b2grad, cost = \
            backpropagation(visibleSize, hiddenSize, numData, W1, W2,
                            b1, b2, trainingSet, weightDecayLambda)
        # already divided by the size of training set

        if all(W1grad == zeros((hiddenSize, visibleSize))) and \
           all(b1grad == zeros((hiddenSize, 1))) and \
           all(W2grad == zeros((visibleSize, hiddenSize))) and \
           all(b2grad == zeros((visibleSize, 1))):
            return W1, W2, b1, b2
        # update the weights
        W1 -= alpha * W1grad + weightDecayLambda * W1
        W2 -= alpha * W2grad + weightDecayLambda * W2
        b1 -= alpha * b1grad.reshape(hiddenSize, 1)
        b2 -= alpha * b2grad.reshape(visibleSize, 1)

        it += 1
        if (it % 50000) == 0:
            print("Cost: %.5f" % cost)
        if it >= maxIteration:
            return W1, W2, b1, b2


def forword(inputData, W1, W2, b1, b2):
    data = array(inputData).reshape(len(inputData), 1)
    # print(("data", data))
    # print(dot(W1, data))
    hd = scipy.special.expit(dot(W1, data)+b1)
    # print(hd)
    out = scipy.special.expit(dot(W2, hd)+b2)
    return out


def test_autoencoder(trainingSet, W1, W2, b1, b2):
    numData = trainingSet.shape[1]
    s = 0
    for i in range(numData):
        data = trainingSet[:, i]
        answer = forword(trainingSet[:, i], W1, W2, b1, b2).transpose()
        diff = sum(sum((answer-data)**2))
        s += diff
    return s/numData


if __name__ == "__main__":
    hiddenSize = 6
    alpha = 0.02  # learning rate
    trainingSet = array([(0.1, 0.2, 0.4, 0.25),
                         (0.2, 0.4, 0.8, 0.5), (0.3, 0.5, 0.9, 0.6)])
    W1, W2, b1, b2 = train(trainingSet, hiddenSize,
                           alpha, weightDecayLambda=0.000001)
                           # maxIteration=10)

    score = test_autoencoder(trainingSet, W1, W2, b1, b2)
    print("Score: %.5f" % (score*1000))


# test_autoencoder error measurement over the training set:
#   without the weight decay term:
#       (hiddenSize = 6, alpha = 0.02)
#       Score: 1.24566
#
#   with the weight decay term:
#       (hiddenSize = 6, alpha = 0.02)
#       when weightDecayLambda=0.0001, the autoencoder performs much worse
#       when weightDecayLambda=0.000001, Score: 1.14421
