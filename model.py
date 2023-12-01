import numpy as np
import pickle
from matplotlib import pyplot as plt


class Layer:
    def __init__(self, input_shape, output_shape, activation):
        self.weights = np.random.randn(input_shape, output_shape) * 0.1
        self.biases = np.zeros((1, output_shape))
        self.activation = activation

    def forward(self, data):
        self.output = np.dot(data, self.weights) + self.biases
        self.output = self.activation(self.output)


class NeuralNetwork:
    def __init__(self, structure):
        self.layers = list()
        self.learning_rate = 1.0
        self.structure = structure

        for i in range(len(structure)-1):
            if i == len(structure) - 2:
                self.layers.append(Layer(structure[i], structure[i+1], self.sigmoid))
            else:
                self.layers.append(Layer(structure[i], structure[i+1], self.relu))

    @staticmethod
    def sigmoid(arr):
        # return 1 / 1 + np.exp(arr)
        return arr

    @staticmethod
    def relu(arr):
        return np.maximum(0, arr)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            return data

    def forward(self, data):
        self.layers[0].forward(data)
        for i in range(len(self.layers)-1):
            self.layers[i+1].forward(self.layers[i].output)

    def train(self, inputs, targets, epochs):
        data_items = inputs.shape[0]
        decay = np.exp(np.log(0.01) / epochs)

        for i in range(epochs):
            if (i+1) % (epochs // 10) == 0:
                print(f"{((i+1)/epochs)*100:.2f}% trained - accuracy: {self.evaluate(inputs, targets)}")

            self.forward(inputs)
            delta_ws = list()
            delta_bs = list()

            j = len(self.layers) - 1

            outputs = self.layers[j].output
            exp_outs = np.exp(outputs)
            probabilities = exp_outs / np.sum(exp_outs, axis=1, keepdims=True)
            gradients = probabilities
            gradients[range(data_items), targets] -= 1
            gradients /= data_items

            while j >= 0:
                if j > 0:
                    prev_layer = self.layers[j-1]
                    delta_ws.insert(0, np.dot(prev_layer.output.T, gradients))
                    delta_bs.insert(0, np.sum(gradients, axis=0, keepdims=True))
                    gradients = np.dot(gradients, self.layers[j].weights.T)
                    gradients[prev_layer.output <= 0] = 0

                else:
                    delta_ws.insert(0, np.dot(inputs.T, gradients))
                    delta_bs.insert(0, np.sum(gradients, axis=0, keepdims=True))

                j -= 1

            for j in range(len(self.layers)-1, -1, -1):
                self.layers[j].weights += -self.learning_rate * delta_ws[j]
                self.layers[j].biases += -self.learning_rate * delta_bs[j]

            self.learning_rate *= decay

    def evaluate(self, inputs, targets):
        self.forward(inputs)
        result = np.mean(np.argmax(self.layers[-1].output, axis=1) == targets)
        return round(result, 3)

    def predict(self, data):
        self.forward(data)
        return self.layers[-1].output

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def decision_boundaries(self, xs, ys):
        h = 0.01
        x_min, x_max = xs[:, 0].min() - 1, xs[:, 0].max() + 1
        y_min, y_max = xs[:, 1].min() - 1, xs[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        z = np.argmax(z, axis=1).reshape(xx.shape)
        plt.contourf(xx, yy, z, cmap=plt.cm.Spectral, alpha=0.8)
        # plt.scatter(xs[:, 0], xs[:, 1], c=ys, s=20, cmap=plt.cm.Spectral)
        plt.show()
