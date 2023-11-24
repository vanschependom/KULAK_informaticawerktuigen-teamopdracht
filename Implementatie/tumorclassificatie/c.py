import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from sklearn import svm

# We laden de data
gegevens = read_csv("data.csv").sample(n=30)

rand = np.random.randint(0, 28)

# We selecteren de kolommen met de features (x-waarden)
trainingFeatures = gegevens.iloc[:, 2:4]
# We selecteren de kolom met de diagnoses (y-waarden)
trainingDiagnoses = gegevens.iloc[:, 1]

trainingX = trainingFeatures.values
trainingY = trainingDiagnoses.values

input1 = trainingFeatures.columns[0]
input2 = trainingFeatures.columns[1]


plt.figure(figsize=(10, 10))

plt.suptitle(
    f"Beslissingsgrenzen voor verschillende waarden van C\n met x1={input1} en x2={input2}")

# De C-term is een metaparameter en staat voor de foutentoleratie van het model.
mogelijkeCs = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]

# Overloop verschillende waarden voor C
for i, Ci in enumerate(mogelijkeCs):

    clf = svm.SVC(kernel='linear', C=Ci)
    clf.fit(trainingX, trainingY)

    voorspelling = clf.predict(trainingX)

    ax1 = plt.subplot(3, 3, i+1)

    # Punten die geclassificeerd worden als goedaardig maar kwaadaardig zijn
    falsePositives = np.where((voorspelling == 'B') & (trainingY == 'M'))
    # Punten die geclassificeerd worden als goedaardig en ook goedaardig zijn
    truePositives = np.where((voorspelling == 'B') & (trainingY == 'B'))
    # Punten die geclassificeerd worden als kwaadaardig maar goedaardig zijn
    falseNegatives = np.where((voorspelling == 'M') & (trainingY == 'B'))
    # Punten die geclassificeerd worden als kwaadaardig en ook kwaadaardig zijn
    trueNegatives = np.where((voorspelling == 'M') & (trainingY == 'M'))

    ax1.scatter(trainingX[trueNegatives, 0], trainingX[trueNegatives, 1], c='r',
                marker='o', label='Kwaadaardig geclassificeerd als "kwaadaardig"')
    ax1.scatter(trainingX[falsePositives, 0], trainingX[falsePositives, 1],
                c='r', marker='x', label='Kwaadaardig geclassificeerd als "goedaardig"')
    ax1.scatter(trainingX[truePositives, 0], trainingX[truePositives, 1],
                c='g', marker='o', label='Goedaardig geclassificeerd als "goedaardig"')
    ax1.scatter(trainingX[falseNegatives, 0], trainingX[falseNegatives, 1],
                c='g', marker='x', label='Goedaarig geclassificeerd als "kwaadaarig"')

    ax1.set_title(f"C={Ci:.1f}")

    # Beslissingslijn plotten
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(
        xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax1.contour(xx, yy, Z, colors='k',
                levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # ax1.set_xlabel(input1)
    # ax1.set_ylabel(input2)

plt.show()
