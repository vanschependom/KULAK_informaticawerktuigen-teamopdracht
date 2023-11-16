from procedures import *
import matplotlib.pyplot as plt
import numpy as np

# Populatiefunctie


# def f(x):
#     return 10**x


# # steekproefgroottes voor k van 0 tem 10
# steekproefgroottes = [20*(2**k) for k in range(10)]

# optimaleKs = list()
# KMSEs = list()
# optimalePs = list()
# PMSEs = list()

# # overloop elke steekproefgrootte
# for steekproefgrootte in steekproefgroottes:

#     trainingX = xsample(steekproefgrootte, 0, 1)
#     trainingY = ysample(trainingX, f)

#     validatieX = xsample(1000, 0, 1)
#     validatieY = ysample(validatieX, f)

#     # KNN REGRESSIE ---------------------------------------------------------------------------------------------
#     kOpt = berekenKOpt(trainingX, trainingY, validatieX, validatieY, True)
#     optimaleKs.append(kOpt)

#     # POLYNOMIALE REGRESSIE -------------------------------------------------------------------------------------
#     pOpt = berekenPOpt(trainingX, trainingY, validatieX, validatieY)
#     optimalePs.append(pOpt)

# # plot de resultaten
# plt.figure(figsize=(10, 8))
# plt.suptitle(
#     "Metaparameters voor KNN-regressie en polynomiale regressie in functie van de steekproefgrootte")
# plt.subplot(2, 1, 1)
# plt.plot(steekproefgroottes, optimaleKs, c='r', label="KNN-regressie")
# plt.xlabel("Steekproefgrootte")
# plt.ylabel("K_opt")
# plt.legend()
# plt.subplot(2, 1, 2)
# plt.plot(steekproefgroottes, optimalePs, c='b', label="Polynomiale regressie")
# plt.xlabel("Steekproefgrootte")
# plt.ylabel("p_opt")
# plt.legend()
# plt.savefig('simulatie7.pdf')

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from time import *

# Populatiefunctie


def f(x):
    return 10**x


steekproefgroottes = [20*(2**k) for k in range(1, 11)]

# hierin gaan we de tijden steken die nodig zijn om de regressielijn te berekenen via de module sklearn
skTijdenLin = list()
skTijdenKnn = list()
# hierin gaan we de tijden steken die nodig zijn om de regressielijn te berekenen via onze eigen OLS procedure
eigenTijdenLin = list()
eigenTijdenKnn = list()

for steekproefgrootte in steekproefgroottes:

    trainingX1 = xsample(steekproefgrootte, 0, 1)
    trainingX2 = xsample(steekproefgrootte, 0, 1)
    trainingY = ysample(trainingX1+trainingX2, f)
    validatieX1 = xsample(steekproefgrootte, 0, 1)
    validatieX2 = xsample(steekproefgrootte, 0, 1)

    columnStack = np.column_stack([trainingX1, trainingX2])
    columnStack1 = np.column_stack(
        [np.ones(len(trainingX1)), trainingX1, trainingX2])
    columnStackVal = np.column_stack([validatieX1, validatieX2])

    # LINEAIRE REGRESSIE -----------------------------------------------------------------------------------------

    # scikit-learn
    t11 = time()
    model = LinearRegression()
    model.fit(columnStack, trainingY)
    b_sk = model.intercept_
    a_sk = model.coef_[0]
    t12 = time()
    # voeg de tijd die nodig was voor het berekenen van de regressielijn toe aan de lijst
    verschilMs = (t12-t11)*1000
    skTijdenLin.append(verschilMs)

    # Eigen OLS regressie
    t21 = time()
    beta = mls(trainingY, columnStack1)
    t22 = time()
    # voeg de tijd die nodig was voor het berekenen van de regressielijn toe aan de lijst
    verschilMs = (t22-t21)*1000
    eigenTijdenLin.append(verschilMs)

    # KNN REGRESSIE ---------------------------------------------------------------------------------------------

    k = 10

    # scikit-learn
    t31 = time()
    model = KNeighborsRegressor(k)
    model.fit(columnStack, trainingY)
    yhat1 = model.predict(columnStackVal)
    t32 = time()
    # voeg de tijd die nodig was voor het berekenen van de regressielijn toe aan de lijst
    verschilS = (t32-t31)
    skTijdenKnn.append(verschilS)

    # Eigen OLS regressie
    t41 = time()
    yhat2 = mknn(columnStack, trainingY, columnStack, k)
    t42 = time()
    # voeg de tijd die nodig was voor het berekenen van de regressielijn toe aan de lijst
    verschilS = (t42-t41)
    eigenTijdenKnn.append(verschilS)


plt.figure(figsize=(10, 6))
plt.suptitle("Verschil in rekentijd tussen sklearn en eigen implementatie")
ax = plt.subplot(1, 2, 1)
ax.set_title("Meervoudige lineaire regressie")
ax.plot(steekproefgroottes, skTijdenLin, c='r', label="sklearn")
ax.plot(steekproefgroottes, eigenTijdenLin, c='b', label="Eigen OLS")
ax.set_xlabel("Steekproefgrootte")
ax.set_ylabel("Tijd (ms)")
ax.legend()
ax = plt.subplot(1, 2, 2)
ax.set_title(f"Meervoudige KNN-regressie (K={k})")
ax.plot(steekproefgroottes, skTijdenKnn, c='r', label="sklearn")
ax.plot(steekproefgroottes, eigenTijdenKnn, c='b', label="Eigen KNN")
ax.set_xlabel("Steekproefgrootte")
ax.set_ylabel("Tijd (s)")
ax.legend()
plt.savefig('simulatie9.2.pdf')
plt.show()
