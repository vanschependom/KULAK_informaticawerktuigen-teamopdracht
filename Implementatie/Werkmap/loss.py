# import matplotlib.pyplot as plt
# import numpy as np

# # Define the loss functions


# def hinge_loss(y_true, y_pred):
#     return np.maximum(0, 1 - y_true * y_pred)


# def logistic_loss(y_true, y_pred):
#     return -np.log(y_pred) if y_true == 1 else -np.log(1 - y_pred)


# # Generate data
# x = np.linspace(-5, 10, 1000)
# y_hinge = hinge_loss(1, x)
# y_logistic = logistic_loss(1, x)

# plt.figure()
# plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# plt.rc('text', usetex=True)
# # Plot the loss functions
# plt.plot(x, y_hinge, label='Hinge Loss')
# plt.plot(x, y_logistic, label='Logistische Loss')
# plt.xlabel(r'$x_i\cdot f(x)$')
# plt.ylabel('Loss waarde')
# plt.legend()
# plt.title('Loss Functions')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # Define the range of x values
# x = np.linspace(-3, 3, 400)

# # Define the loss functions
# zero_one_loss = np.where(x < 0, 1, 0)
# logistic_loss = -np.log(1/(1 + np.exp(-x)))
# hinge_loss = np.where(1 - x < 0, 0, 1 - x)

# # Plot the loss functions
# plt.figure(figsize=(6, 4))
# plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# plt.rc('text', usetex=True)
# plt.plot(x, zero_one_loss, 'k-', label=r'\textit{Zero-one loss}')
# plt.plot(x, logistic_loss, 'r-', label=r'\textit{Logistic loss}, zoals bij LR')
# plt.plot(x, hinge_loss, 'b-', label=r'\textit{Hinge loss}, zoals bij SVM')
# plt.xlabel(r'$y_i \cdot \hat{f}(x_i)$')
# plt.ylabel('Loss')
# plt.legend(loc='upper right')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('lossfuncties.pdf')
# plt.show()from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from mpl_toolkits.mplot3d import Axes3D

iris = datasets.load_iris()
X = iris.data[:, :3]  # we only take the first three features.
Y = iris.target

# make it binary classification problem
X = X[np.logical_or(Y == 0, Y == 1)]
Y = Y[np.logical_or(Y == 0, Y == 1)]

model = svm.SVC(kernel='linear')
clf = model.fit(X, Y)

# The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
# Solve for w3 (z)


def z(x, y): return (-clf.intercept_[0]-clf.coef_[0]
                     [0]*x - clf.coef_[0][1]*y) / clf.coef_[0][2]


xlim = [4.0, 8.0]
ylim = [1.5, 4.5]

x, y = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                   np.linspace(ylim[0], ylim[1], 50))

fig = plt.figure()
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
plt.rc('text', usetex=True)
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(X[Y == 0, 0], X[Y == 0, 1], X[Y == 0, 2],
          'og', label='Groep 1', markeredgecolor='black')
ax.plot3D(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2],
          'sr', label='Groep 2', markeredgecolor='black')
ax.plot_surface(x, y, z(x, y), color='blue', alpha=0.5)
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$x_3$')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.view_init(30, 60)
plt.tight_layout()
plt.legend(loc='upper right')
plt.savefig('3d.pdf')
plt.show()
