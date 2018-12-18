
import typing
import pprint
import numpy as np


class Layer():

    def __init__(self, input, output):
        self.__weights: np.matrix = np.matrix(
            np.random.uniform(
                low=-1,
                high=1,
                size=(input, output)))

    def getMatrix(self):
        return self.__weights

    def feedFoward(self, input):
        return input * self.__weights

    def error(self, target, output):
        return target.transpose() - output.transpose()

    def hidden_error(self, error):
        return self.__weights * error

    def sigm(self, x):
        return 1. / (1. + np.exp(-x))

    def dsigm(self, x):
        return self.sigm(x) * (1. - self.sigm(x)).transpose()

    def backpropagation(self, input, error, learning_rate):
        self.__weights += (learning_rate * error * self.dsigm(self.feedFoward(input)) * input).transpose()
        # self.__weights = self.__weights.add(error.mul(learning_rate).mul(input).transpose())
        return self.hidden_error(error)


learning_rate = 0.000001

input = Layer(2, 2048)
output = Layer(2048, 1)

for _ in range(100):
    iff = input.feedFoward(np.matrix([1, 1]))
    # pprint.pprint(type(iff))
    off = output.feedFoward(iff)
    error = output.error(np.matrix([1]), off)
    pprint.pprint(off)
    # pprint.pprint(error)
    # pprint.pprint(output.hidden_error(error))
    # pprint.pprint(output.getMatrix())

    input.backpropagation(np.matrix([1, 1]), output.backpropagation(iff, error, learning_rate), learning_rate)
    print("")
# iff = input.feedFoward(np.matrix([1, 1]))
# # pprint.pprint(type(iff))
# off = output.feedFoward(iff)
# error = output.error(np.matrix([1, 0]), off)
# pprint.pprint(error)
# output.backpropagation(off, output.hidden_error(error), learning_rate)
