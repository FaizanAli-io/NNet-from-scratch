from model import NeuralNetwork
from data import *

model = NeuralNetwork((2, 64, 64, 4))
x, y = boxed_dataset(400, 4, 0.25)
show_data(x, y)

model.train(x, y, 2000)
print(model.evaluate(x, y))
model.decision_boundaries(x, y)
