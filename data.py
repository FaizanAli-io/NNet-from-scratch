import numpy as np
from matplotlib import pyplot as plt


def spiral_dataset(points, classes, s, noise):
    x = np.zeros((points * classes, 2))
    y = np.zeros((points * classes), dtype="uint8")
    for k in range(classes):
        ix = range(points * k, points * (k+1))
        r = np.linspace(0, 1, points)
        t = np.linspace(k * 4, (k + 1) * 4, points) + (np.random.randn(points) * noise)
        x[ix] = np.c_[r * np.cos(t * s), r * np.sin(t * s)]
        y[ix] = k
    return x, y


def boxed_dataset(points, classes, noise):
    x = np.zeros((points * classes ** 2, 2))
    y = np.zeros((points * classes ** 2), dtype="uint8")
    xx, yy = np.meshgrid(np.arange(0, classes), np.arange(0, classes))
    xx, yy = xx.ravel(), yy.ravel()
    for k in range(classes ** 2):
        ix = range(points * k, points * (k + 1))
        x[ix] = np.c_[xx[k] + np.random.randn(points) * noise, yy[k] + np.random.randn(points) * noise]
        y[ix] = (k + k // classes) % classes
    return x, y


def circular_dataset(points, classes, noise):
    x = np.zeros((points * classes, 2))
    y = np.zeros((points * classes), dtype="uint8")
    for k in range(classes):
        ix = range(points * k, points * (k + 1))
        r = k + np.random.randn(points) * noise
        t = np.linspace(0, 2 * np.pi, points) + np.random.randn(points)
        x[ix] = np.c_[r * np.cos(t), r * np.sin(t)]
        y[ix] = k
    return x, y


def show_data(xs, ys):
    plt.scatter(xs[:, 0], xs[:, 1], c=ys, s=20)
    plt.show()
