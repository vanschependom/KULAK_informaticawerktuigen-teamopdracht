#
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn import datasets
# data opsplitsen in trainings-, validatie- en testset
from sklearn.model_selection import train_test_split
# grafieken
from matplotlib import pyplot as plt
# nodig voor het maken van het svm model


def fit(X, y, leerTempo, lambdaParam, aantalIteraties):
    '''
    Deze functie traint een SVM model.

    Parameters
    ----------
    X : ndarray
        De datapunten, elk met twee features.
    y : ndarray
        De labels van de datapunten. Deze zijn -1 of 1.
    leerTempo : float
        Het leerTempo van het model. Dit is een getal tussen 0 en 1.
    lambdaParam : float
        De regularisatie parameter van het model.

    Returns
    -------
    w : ndarray
        De gewichtsvector van het model. Deze vector heeft dezelfde lengte als het aantal features.
    b : float
        De bias term van het model.
    '''
    aantalPunten, aantalFeatures = X.shape

    # we willen een vector w met dezelfde lengte als het aantal features
    w = np.zeros(aantalFeatures)

    # we willen een bias term b
    b = 0

    for x in range(aantalIteraties):

        for i, x_i in enumerate(X):

            # y_i(x_i * w - b) >= 1
            goedGeclassificeerd = y[i] * (np.dot(x_i, w) - b) >= 1

            if goedGeclassificeerd:

                # verander de gewichtsvector w
                w -= leerTempo * (2 * lambdaParam * w)

                # b blijft onveranderd

            # als the voorspelde waarde niet hetzelfde is als de echte waarde
            # trek de loss functie af van de gewichtsvector w
            else:

                w -= leerTempo * (2 * lambdaParam *
                                  w - np.dot(y[i], x_i))
                b -= leerTempo * y[i]

    return w, b


def voorspelling(X, w, b):
    '''
    Deze functie voorspelt de labels van de datapunten met behulp van het SVM model.

    Parameters
    ----------
    X : ndarray
        De datapunten, elk met twee features.
    w : ndarray
        De gewichtsvector van het model.
    b : float
        De bias term van het model.
    '''
    voorspelling = np.dot(X, w) - b

    # als de voorspelling boven de beslissingsgrens ligt, dan is de voorspelling 1, anders -1
    return np.sign(voorspelling)


def hyperplaneFunctie(x, w, b, offset):
    '''
    Deze functie returned de y-waarde op het hyperplane voor een gegeven x-waarde.

    Parameters
    ----------
    x : float
        De x-waarde waarvoor de y-waarde op het hyperplane wordt berekend.
    w : ndarray
        De gewichtsvector van het model.
    b : float
        De bias term van het model.
    offset : float
        De offset van de hyperplane.

    Returns
    -------
    float
        De y-waarde op het hyperplane voor de gegeven x-waarde.
    '''
    return (-w[0] * x + b + offset) / w[1]


def accuraatheid(voorspellingen, labels, w, b):
    '''
    Deze functie berekent de accuraatheid van het model.

    Parameters
    ----------
    voorspellingen : ndarray
        De voorspellingen van het model.
    labels : ndarray
        De labels van de datapunten.
    w : ndarray
        De gewichtsvector van het model.
    b : float
        De bias term van het model.

    Returns
    -------
    float
        De accuraatheid van het model in procent.
    '''
    aantalCorrect = 0
    for i in range(len(voorspellingen)):
        if voorspellingen[i] == labels[i]:
            aantalCorrect += 1
    return (aantalCorrect / len(voorspellingen)) * 100
