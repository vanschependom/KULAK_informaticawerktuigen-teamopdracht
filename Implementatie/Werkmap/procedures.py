# Hierin mogen enkel def-statements (definities van functies) komen.
# Er mogen dus geen commando's effectief uitgevoerd worden en er mag geen output gegenereerd worden.
# Op andere plaatsen kunnen we deze functies dan aanroepen indien we deze file importeren met 'import procedures'.
'''
Functies moeten van onderstaande vorm zijn:

def functie (x,y):
    """
    Omschrijf het doel van de functie

    Parameters
    ----------
    x : datatype
    Omschrijf de rol van de parameter x
    y : datatype
    Omschrijf de rol van de parameter y

    Returns
    -------
    datatype
    Omschrijf de betekenis van de output
    
    To do
    -----
    Nog te implementeren , geeft voorlopig gewoon de som terug
    """
    out = x + y
    return out

Je kan bij het aanmaken van een nieuwe functie dus bovenstaande snippet kopiëren.
'''

import numpy as np


def xsample(n, a=0, b=1):
    """
    Deze functie genereert een vector (ndarray) van lengte n met uniform verdeelde getallen tussen a en b.

    Parameters
    ----------
    n : int
        Lengte van de vector
    a : int, optioneel
        Beginwaarde, standaardwaarde is 0
    b : int, optioneel
        Eindwaarde, standaardwaarde is 1

    Returns
    -------
    ndarray
         een vector van lengte n met uniform verdeelde getallen tussen a en b.
    """
    # return een vector met willekeurige waarden van a tot en met b
    return np.random.random_sample(n)*(b-a)+a


def ysample(x, f, e=1):
    """
    Deze functie genereert een vector (ndarray) van dezelfde lengte als vector x volgens de formule
    y=f(x)+r met r normaal verdeelde getallen met gemiddelde nul en standaarddeviatie e.

    Parameters
    ----------
    x : ndarray
        Een vector met x-waarden
    f : function
        De functie populatiefunctie f
    e : int, optioneel
        Standaarddeviatie

    Returns
    -------
    ndarray
        Een vector volgens de formule y=f(x)+r met normaal verdeelde getallen met gemiddelde 0 en standaarddeviatie e.
    """

    # de residus r die gemiddeldes 0 hebben en standaarddeviatie e in vectorvorm
    residus = np.array(np.random.normal(loc=0, scale=e, size=len(x)))
    # de functiewaarden f(x) voor alle x_i in x in vectorvorm
    functiewaarden = f(x)
    # tel de residus op bij de functiewaarden en return ze
    return (functiewaarden + residus)


def ols(y, x):
    """
    Deze functie berekent de Ordinary Least Squares (OLS) regressierechte bij gegeven vectoren x en y.

    Parameters
    ----------
    y : ndarray
        Een vector met y-waarden
    x : ndarray
        Een vector met x-waarden

    Returns
    -------
    tuple
        Een koppel (a,b) zodat y=a*x+b de OLS-reggressierechte is bij gegeven vectoren x en y (ndarray). Hierbij komt a overeen met β_0 en b met β_1.
    """

    # zie pagina 71 van ISL voor de formules van Beta1 en Beta2
    # hier is β1 = a en β0 = b

    # bereken de gemiddelden van x en y
    gem_x = np.mean(x)
    gem_y = np.mean(y)

    # bereken β1 door de formule van pagina 71 van ISL toe te passen
    # we voegen een sommatie toe over alle x_i in x
    a = np.sum((x - gem_x) * (y - gem_y)) / np.sum((x - gem_x) ** 2)
    # bereken β0 door de formule van pagina 71 van ISL toe te passen
    b = gem_y - a * gem_x

    # return het koppel (a,b)
    return (a, b)


def knn(x0, y, x, k):
    """
    Deze functie voor enkelvoudige regressie berekent de voorspelling(en) in x0 volgens KNN-regressie bij gegeven regressor x en respons y (ndarray).

    Parameters
    ----------
    x0 : ndarray
        Een vector van x-waarden waarvoor we de voorspelling willen berekenen
    y : ndarray
        Een vector van responsen (outputs, y-waarden)
    x : ndarray
        Een vector van regressoren (inputs, x-waarden)
    k : int
        Het aantal nabijgelegen buren dat gebruikt wordt in de KNN-regressie

    Returns
    -------
    ndarray
        Een vector van voorspellingen
    """

    # initialiseer een lege vector om de voorspellingen in op te slaan
    voorspellingen = np.empty(len(x0))

    # voor elk punt x0_i in de vector x0
    for i, x0_i in enumerate(x0):
        # bereken de afstand tussen x0_i en alle punten in x
        afstanden = np.abs(x - x0_i)
        # sorteer de afstanden en houd de indices bij
        gesorteerdeIndices = np.argsort(afstanden)
        # selecteer de indices van de k dichtstbijzijnde punten
        kDichtstbijzijndeIndices = gesorteerdeIndices[:k]
        # bereken de gemiddelde gemiddelde y-waarde van de k dichtstbijzijnde punten
        # zet deze gemiddelde waarde gelijk aan de voorspelling voor de waarde x0_i behorend tot de vector x0
        voorspellingen[i] = np.mean(y[kDichtstbijzijndeIndices])

    # return de voorspellingen
    return voorspellingen


def mknn(x0, y, x, k):
    """
    Deze functie voor meervoudige regressie berekent de voorspelling(en) in x0 volgens KNN-regressie bij gegeven matrix regressoren x (met meerdere x-waarden) en respons y (ndarray).

    Parameters
    ----------
    x0 : ndarray
        Een vector van x-waarden waarvoor we de voorspelling willen berekenen
    y : ndarray
        Een vector van responsen (outputs, y-waarden)
    x : ndarray
        Een matrix van regressoren (inputs, x-waarden)
    k : int
        Het aantal nabijgelegen buren dat gebruikt wordt in de KNN-regressie

    Returns
    -------
    ndarray
        Een vector van voorspellingen
    """

    # initialiseer een lege vector om de voorspellingen in op te slaan
    voorspellingen = np.array([])

    for x0_i in x0:

        # bereken de afstanden tussen x0_i en alle x_i in x
        afstanden = np.linalg.norm(x0_i - x, axis=1)

        # sorteer de afstanden van klein naar groot en onthoud de indices
        gesorteerdeIndices = np.argsort(afstanden)

        # neem de k dichtstbijzijnde indices
        kDichtstbijzijndeIndices = gesorteerdeIndices[:k]

        # bereken de voorspelling voor x0_i
        voorspelling = np.mean(y[kDichtstbijzijndeIndices])

        # voeg de voorspelling toe aan de vector van voorspellingen
        voorspellingen = np.append(voorspellingen, voorspelling)

    return voorspellingen


def mls(y, X):
    """
    Deze functie genereert een vector beta zodat y=X*beta het kleinste kwadratenhypervlak is bij gegeven
    matrix X en responsvector Y (ndarray).

    Parameters
    ----------
    y : ndarray
        Een vector van responsen (outputs, y-waarden)
    X : ndarray
        Een matrix van regressoren (inputs, x-waarden) met meerdere x'en (x1,x2,...)

    Returns
    -------
    ndarray
        Een vector beta zodat y=X*beta het kleinste kwadratenhypervlak is bij gegeven matrix X en responsvector Y (ndarray).
    """

    # voeg een kolom met eentjes toe voor de X-matrix
    # X1 = np.column_stack((np.ones(len(X)), X))

    # bereken beta
    X_T = np.transpose(X)
    X_TXinv = np.linalg.inv(np.dot(X_T, X))
    X_TXinvX_T = np.dot(X_TXinv, X_T)
    beta = np.dot((X_TXinvX_T), y)

    return beta


def berekenKOpt(trainingX, trainingY, validatieX, validatieY, enkelvoudig):
    """
    Deze functie berekent de optimale K-waarde voor KNN-regressie.

    Parameters
    ----------
    trainingX : ndarray
        Een vector/matrix van regressoren (inputs, x-waarden) waarop de KNN-regressie wordt getraind
    trainingY : ndarray
        Een vector van responsen (outputs, y-waarden) waarop de KNN-regressie wordt getraind
    validatieX : ndarray
        Een vector/matrix van regressoren (inputs, x-waarden) waarop de KNN-regressie wordt getest
    validatieY : ndarray
        Een vector van responsen (outputs, y-waarden) waarop de KNN-regressie wordt getest


    Returns
    -------
    int
        De optimale waarde voor K
    """

    # mogelijke k-waarden, van 1 tem 20
    mogelijkeKWaarden = [i for i in range(1, len(trainingX))]

    # de lijst met alle MSE's
    validatieMSEs = list()

    kOpt = 0

    # overloop elke mogelijke waarde van k
    for k in mogelijkeKWaarden:

        if enkelvoudig:
            # f hoedje
            validatieVoorspelling = knn(validatieX, trainingY, trainingX, k)
        else:
            validatieVoorspelling = mknn(validatieX, trainingY, trainingX, k)

        # y_i - f^
        validatieVerschil = validatieY - validatieVoorspelling

        # MSE = 1/n * som_i=1->n ((y_i-f^)**2)
        validatieMSE = np.mean(validatieVerschil**2)

        # voeg de validatie MSE toe aan de lijst met validatie MSE's
        validatieMSEs.append(validatieMSE)

        # als de validatie MSE kleiner is dan de tot nu toe kleinste, verander Kopt
        if validatieMSE <= min(validatieMSEs):
            kOpt = k

    return kOpt


def berekenPOpt(trainingX, trainingY, validatieX, validatieY):
    """
    Deze functie berekent de optimale p-waarde voor polynomiale regressie.

    Parameters
    ----------
    trainingX : ndarray
        Een vector/matrix van regressoren (inputs, x-waarden) waarop de KNN-regressie wordt getraind
    trainingY : ndarray
        Een vector van responsen (outputs, y-waarden) waarop de KNN-regressie wordt getraind
    validatieX : ndarray
        Een vector/matrix van regressoren (inputs, x-waarden) waarop de KNN-regressie wordt getest
    validatieY : ndarray
        Een vector van responsen (outputs, y-waarden) waarop de KNN-regressie wordt getest


    Returns
    -------
    int
        De optimale waarde voor p
    """

    mogelijkeGraden = [i for i in range(1, len(trainingX))]

    # de lijst met alle MSE's
    validatieMSEs = list()

    pOpt = 1

    # overloop elke graad p
    for p in mogelijkeGraden:

        # creëer de gegevensmatrix X met trainingsdata
        X = np.column_stack([trainingX**i for i in range(p+1)])

        # bereken de beta vector
        beta = mls(trainingY, X)

        # bereken de voorspellingen voor de validatie data
        validatieVoorspelling = np.dot(np.column_stack(
            [validatieX**i for i in range(len(beta))]), beta)

        # bereken de MSE voor de validatie data
        validatieMSE = np.mean((validatieY - validatieVoorspelling)**2)
        # voeg de MSE van de validatie data toe aan de lijst
        validatieMSEs.append(validatieMSE)

        # kijk of de MSE van de validatie data kleiner is dan de kleinste MSE tot nu toe
        if validatieMSE <= min(validatieMSEs):
            pOpt = p

    return pOpt


def coefficienten(beta, xx):
    '''
    Deze functie berekent alle coefficienten van een polynomiale regressie.

    Parameters
    ----------
    beta: ndarray
        De vector beta
    xx: ndarray
        De vector met x-waarden

    Returns
    -------
    ndarray
        De matrix met alle coefficienten van de polynomiale regressie
    '''
    return np.column_stack([xx**i for i in range(len(beta))])
