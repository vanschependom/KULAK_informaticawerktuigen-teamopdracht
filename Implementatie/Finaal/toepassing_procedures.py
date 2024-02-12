#
import numpy as np


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
        De afstand van de beslissingsrechte tot de oorsprong
    '''
    aantalPunten, aantalFeatures = X.shape

    # we willen een vector w met dezelfde lengte als het aantal features
    w = np.zeros(aantalFeatures)

    b = 0

    for x in range(aantalIteraties):

        # overloop alle punten
        for i, x_i in enumerate(X):

            # Goed geclassificeerd als y_i(x_i * w - b) >= 1
            goedGeclassificeerd = y[i] * (np.dot(x_i, w) - b) >= 1

            if goedGeclassificeerd:

                # verander de gewichtsvector w
                w -= leerTempo * (2 * lambdaParam * w)

                # b blijft onveranderd

            # als het punt in de marge ligt op verkeerd geclassificeerd werd
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
        De afstand van de beslissingsrechte tot de oorsprong
    '''
    voorspelling = np.dot(X, w) - b

    # als de voorspelling boven de beslissingsgrens ligt, dan is de voorspelling 1, anders -1
    # hiervoor gebruiken we een sign functie
    return np.sign(voorspelling)


def hyperplaneFunctie(x, w, b, offset):
    '''
    Deze functie returnt de y-waarde op het hyperplane voor een gegeven x-waarde.
    Dit hebben we nodig om de hyperplanes te plotten.

    Parameters
    ----------
    x : float
        De x-waarde waarvoor de y-waarde op het hyperplane wordt berekend.
    w : ndarray
        De gewichtsvector van het model.
    b : float
        De afstand van de beslissingsrechte tot de oorsprong
    offset : float
        De offset van de hyperplane. (-1, 0 of 1)

    Returns
    -------
    float
        De y-waarde op het hyperplane voor de gegeven x-waarde.
    '''
    return (-w[0] * x + b + offset) / w[1]


def accuraatheid(voorspellingen, labels):
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
        De afstand van de beslissingsrechte tot de oorsprong

    Returns
    -------
    float
        De accuraatheid van het model in procent.
    '''
    aantalCorrect = 0
    # overloop elk datapunt
    for i in range(len(voorspellingen)):
        # als de voorspelling voor het datapunt overeenkomt met het effectieve label, is de voorspelling correct
        if voorspellingen[i] == labels[i]:
            # verhoog het aantal correcte voorspellingen met 1
            aantalCorrect += 1
    # return een percentage
    return (aantalCorrect / len(voorspellingen)) * 100
