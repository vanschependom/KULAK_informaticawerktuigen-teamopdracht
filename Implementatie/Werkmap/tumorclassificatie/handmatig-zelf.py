#
import numpy as np
import matplotlib.pyplot as plt


def main():

    X = genereerPunten()
    y = berekenKlasse(X)

    mogelijkeCWaarden = [1]

    for C in mogelijkeCWaarden:
        w = trainSVM(X, y, C)
        plt.figure()
        plot_decision_boundary(X, y, w, C)


def plot_decision_boundary(X, y, W, C):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired,
                edgecolors='k', marker='o')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(np.linspace(
        xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    Z = (np.c_[xx.ravel(), yy.ravel()]).dot(W)
    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, colors='k',
                levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.title(f'SVM Decision Boundary (C={C})')
    plt.show()


def genereerPunten():

    # random logica
    randomSeed = np.random.randint(1, 100)
    np.random.seed(randomSeed)

    # genereer 100 datapunten met 2 inputs of x-waarden
    X = np.random.randn(30, 2)

    return X


def groterDanNul(x):
    # indien x groter is dan 0, dan is de output 1, anders 0
    return (x > 0).astype(int)


def berekenKlasse(X):

    # indien de som groter is dan 0, dan is de output (klasse) 1, anders -1
    y = groterDanNul(X[:, 0] + X[:, 1]) * 2 - 1

    return y


def berekenHingeLoss(w, X, y):

    # bereken de hinge loss
    # als een punt correct geclassificeerd is, dan is de hinge loss 0
    # als een punt niet correct geclassificeerd is, dan is de hinge loss 1 - y * (X.dot(w)), ofwel de afstand tot de marge
    hingeLoss = 1 - y * (X.dot(w))

    return hingeLoss


def berekenVerlies(w, X, y, C):

    # bereken de hinge loss
    hingeLoss = berekenHingeLoss(w, X, y)

    # indien de hinge loss kleiner is dan 0, stel hem dan in op 0
    hingeLoss[hingeLoss < 0] = 0

    # bereken de regularisatie term
    # deze term wordt toegevoegd om overfitting te voorkomen
    # Ddor deze term aan de verliesfunctie toe te voegen, worden de gewichten van het model bestraft als ze te groot worden
    # er zijn verschillende regularisatie termen mogelijk, maar de L2 regularisatie is het meest gebruikelijk
    # we stellen daarbij de term op 0.5 * de som van de kwadraten van de gewichten
    regularisatieTerm = 0.5 * np.sum(w**2)

    # bereken het verlies
    # het voorschrift van de verliesfunctie is C * hingeLoss + regularisatieTerm
    # het verlies is de som van de hinge loss en de regularisatie term
    # het is een maat voor de kwaliteit van het model
    verlies = C * np.sum(hingeLoss) + regularisatieTerm

    return verlies


def berekenGradient(w, X, y, C):

    hingeLoss = berekenHingeLoss(w, X, y)

    # de gradient is de afgeleide van de verliesfunctie
    # het voorschrift van de verliesfunctie is C * hingeLoss + regularisatieTerm
    # de afgeleide van de hinge loss is -y * X
    # de afgeleide van de regularisatie term is w
    # de afgeleide van de verliesfunctie is dus C * -y * X + w
    gradient = C * np.dot(-y * groterDanNul(hingeLoss), X) + w

    return gradient


def trainSVM(X, y, C, snelheid=0.01, aantalTijdsperiodes=1000):

    # initialiseer de gewichtsvector w met nullen
    w = np.zeros(X.shape[1])

    # train de SVM
    for tijdsperiode in range(aantalTijdsperiodes):
        # bereken de kwaliteit van het model
        verlies = berekenVerlies(w, X, y, C)
        # bereken de gradient
        # dit is de afgeleide van de verliesfunctie
        gradient = berekenGradient(w, X, y, C)
        # de gewichten worden veranderd in de richting van de sterkste daling van het verlies
        w -= 0.01 * gradient

        if tijdsperiode % 100 == 0:
            print(f'Tijdsperiode {tijdsperiode}, Verlies: {verlies}')

            # Plot de beslissingsgrens en huidige gewichtsvector
            plot_decision_boundary(X, y, w, C)

    return w


main()
