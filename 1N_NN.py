from numpy import exp, array, random, dot

# BASED OFF OF WORK FROM MIT_COURSEWARE
class NeuralNetwork():
    def __init__(self):
        self.weights = 2 * random.random((3, 1)) - 1

    def __sigmoid(self, x):
        return 1.0 / (1.0 + exp(-x))

    def __sig_deriv(self, x):
        return x * (1.0 - x)

    def train(self, inpTraining, outTraining, num):
        for i in xrange(num):
            output = self.evaluate(inpTraining)
            error = outTraining - output
            adjustment = dot(inpTraining.T, error * self.__sig_deriv(output))
            self.weights += adjustment

    def evaluate(self, inputs):
        return self.__sigmoid(dot(inputs, self.weights))


if __name__ == "__main__":

    network = NeuralNetwork()

    print "Random starting weights: "
    print network.weights

    inpTraining = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0]])
    outTraining = array([[1, 1, 0, 0, 0]]).T

    network.train(inpTraining, outTraining, 10000)

    print "Synaptic weights after training: "
    print network.weights

    print "Predictions for [1, 0, 0]:  "
    print network.evaluate(array([1, 0, 0]))