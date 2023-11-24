import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn import datasets
# data opsplitsen in trainings-, validatie- en testset
from sklearn.model_selection import train_test_split
# grafieken
from matplotlib import pyplot as plt
# nodig voor het maken van het svm model


def fit(X, y, leerTempo, c, aantalIteraties):

    aantalPunten, aantalFeatures = X.shape

    # # we willen twee klasses: punten met y=1 en punten met y=-1
    # y_ = np.where(y <= 0, -1, 1)

    # we willen een vector w met dezelfde lengte als het aantal features
    w = np.zeros(aantalFeatures)

    # we willen een bias term b
    b = 0

    for x in range(aantalIteraties):

        for i, x_i in enumerate(X):

            # y_i(x_i * w - b) >= 1
            voorwaardeVocaan = y[i] * (np.dot(x_i, w) - b) >= 1

            if voorwaardeVocaan:

                # verander de gewichtsvector w
                w -= leerTempo * (2 * c * w)

            else:

                w -= leerTempo * (2 * c * w - np.dot(x_i, y[i]))
                b -= leerTempo * y[i]

    return w, b


def voorspelling(X, w, b):

    voorspelling = np.dot(X, w) - b

    # als de voorspelling boven het hyperplane ligt, dan is de voorspelling 1, anders -1
    return np.sign(voorspelling)


def hyperplaneFunctie(x, w, b, intercept):
    return (-w[0] * x + b + intercept) / w[1]


X, y = datasets.make_blobs(
    n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
)
y = np.where(y == 0, -1, 1)

trainingX, testX, trainingY, testY = train_test_split(
    X, y, test_size=0.2, random_state=123
)

leerTempo = 0.001
c = 1
aantalIteraties = 10000

w, b = fit(trainingX, trainingY, leerTempo, c, aantalIteraties)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(trainingX[:, 0], trainingX[:, 1], marker="o", c=trainingY)

x0_1 = min(trainingX[:, 0])
x0_2 = max(trainingX[:, 0])

x1_1 = hyperplaneFunctie(x0_1, w, b, 0)
x1_2 = hyperplaneFunctie(x0_2, w, b, 0)

x1_1_m = hyperplaneFunctie(x0_1, w, b, -1)
x1_2_m = hyperplaneFunctie(x0_2, w, b, -1)

x1_1_p = hyperplaneFunctie(x0_1, w, b, 1)
x1_2_p = hyperplaneFunctie(x0_2, w, b, 1)

ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

x1_min = np.amin(trainingX[:, 1])
x1_max = np.amax(trainingX[:, 1])
ax.set_ylim([x1_min - 3, x1_max + 3])

plt.show()
