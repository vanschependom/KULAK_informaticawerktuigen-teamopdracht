# from sklearn import datasets, linear_model
# from procedures import *
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import rc

# trainingX = xsample(70, 0, 1)


# def f(x):
#     return x


# trainingY = ysample(trainingX, f, e=0.1)

# # bereken de helling a en de intercept b van de regressielijn
# a, b = ols(trainingY, trainingX)


# def geschatteFunctie(x):
#     return a*x + b


# classificatieX, classificatieY = datasets.make_blobs(
#     n_samples=50, n_features=2, centers=2
# )

# x1 = classificatieX[:, 0]
# x2 = classificatieX[:, 1]

# # maak een lineaire classifier
# clf = linear_model.LogisticRegression()
# clf.fit(classificatieX, classificatieY)

# # bereken de helling a en de intercept b van de regressielijn
# a2 = clf.coef_[0][0]
# b2 = clf.coef_[0][1]


# def classifier(x):
#     return -(a2/b2)*x - clf.intercept_[0]/b2


# x_min = np.min(classificatieX[:, 0])
# x_max = np.max(classificatieX[:, 0])

# xx = np.linspace(x_min, x_max, 100)

# plt.figure(figsize=(10, 5))

# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# rc('text', usetex=True)

# ax = plt.subplot(1, 2, 1)
# ax.set_title("Regressie")
# ax.scatter(trainingX, trainingY, c="k", s=10, label="Datapunten")
# ax.plot(trainingX, geschatteFunctie(trainingX), c="b", label="Regressierechte")
# # plot de afwijking
# # for i in range(len(trainingX)):
# #     ax.plot([trainingX[i], trainingX[i]], [
# #             trainingY[i], geschatteFunctie(trainingX[i])], "k--", lw=0.5)
# ax.legend()
# ax.set_xlabel(r'$x$')
# ax.set_ylabel(r'$y$')

# groep1 = classificatieX[classificatieY == 0]
# groep2 = classificatieX[classificatieY == 1]

# groep1x1 = groep1[:, 0]
# groep1x2 = groep1[:, 1]
# groep2x1 = groep2[:, 0]
# groep2x2 = groep2[:, 1]

# ax = plt.subplot(1, 2, 2)
# ax.set_title("Classificatie")
# ax.scatter(groep1x1, groep1x2, c="r", s=10, label="Groep 1")
# ax.scatter(groep2x1, groep2x2, c="g", s=10, label="Groep 2")
# ax.plot(xx, classifier(xx), label="Beslissingsgrens", c="k")
# ax.legend()
# ax.set_xlabel(r'$x_1$')
# ax.set_ylabel(r'$x_2$')

# plt.savefig("regressievsclassificatie.pdf")
# plt.show()
