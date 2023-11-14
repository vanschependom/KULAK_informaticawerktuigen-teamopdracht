from procedures import *
import matplotlib.pyplot as plt
import numpy as np

# Populatiefunctie


def f(x):
    return 10**x


# steekproefgroottes voor k van 0 tem 10
steekproefgroottes = [20*2**k for k in range(7)]

optimaleKs = list()
KMSEs = list()
optimalePs = list()
PMSEs = list()

# overloop elke steekproefgrootte
for steekproefgrootte in steekproefgroottes:

    trainingX = xsample(steekproefgrootte, 0, 1)
    trainingY = ysample(trainingX, f)

    validatieX = xsample(1000, 0, 1)
    validatieY = ysample(validatieX, f)

    # KNN REGRESSIE ---------------------------------------------------------------------------------------------
    kOpt = berekenKOpt(trainingX, trainingY, validatieX, validatieY, True)
    optimaleKs.append(kOpt)

    # POLYNOMIALE REGRESSIE -------------------------------------------------------------------------------------
    pOpt = berekenPOpt(trainingX, trainingY, validatieX, validatieY)
    optimalePs.append(pOpt)

# plot de resultaten
plt.figure(figsize=(10, 8))
plt.suptitle(
    "Metaparameters voor KNN-regressie en polynomiale regressie in functie van de steekproefgrootte")
plt.subplot(2, 1, 1)
plt.plot(steekproefgroottes, optimaleKs, c='r', label="KNN-regressie")
plt.xlabel("Steekproefgrootte")
plt.ylabel("K_opt")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(steekproefgroottes, optimalePs, c='b', label="Polynomiale regressie")
plt.xlabel("Steekproefgrootte")
plt.ylabel("p_opt")
plt.legend()
plt.savefig('simulatie7.pdf')
