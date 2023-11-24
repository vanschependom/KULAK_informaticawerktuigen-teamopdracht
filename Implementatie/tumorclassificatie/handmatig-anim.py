import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Genereer wat voorbeeldgegevens
np.random.seed(42)
data = np.random.randn(30, 2)
labels = np.where(data[:, 0] + data[:, 1] > 0, 1, -1)

# Voeg een intercept toe aan de gegevens
data_with_intercept = np.c_[np.ones((data.shape[0], 1)), data]

# Initialisatie van de SVM-parameters
weights = np.random.randn(data_with_intercept.shape[1])
learning_rate = 0.01
epochs = 100

# Hulpfunctie voor de lineaire SVM-classificatie


def svm_classification(data, weights):
    return np.sign(np.dot(data, weights))

# Hulpfunctie voor het updaten van de SVM-parameters met behulp van SGD


def svm_sgd_update(data, labels, weights, learning_rate):
    errors = labels - svm_classification(data, weights)
    gradient = -np.dot(errors, data) / len(data)
    weights -= learning_rate * gradient
    return weights


# Initialisatie van de plot
fig, ax = plt.subplots()
scatter = ax.scatter(data[:, 0], data[:, 1], c=labels,
                     cmap=plt.cm.Paired, marker='o', s=50)

# Animatiefunctie voor het bijwerken van de plot


def update(frame):
    global weights
    ax.clear()

    # Voer SGD-update uit voor elke epoch
    for epoch in range(epochs):
        weights = svm_sgd_update(
            data_with_intercept, labels, weights, learning_rate)

    # Plaats de gewichtsvector in de plot
    x_vals = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
    y_vals = -(weights[0] + weights[1] * x_vals) / weights[2]
    ax.plot(x_vals, y_vals, 'k-', label='Gewichtsvector')

    # Markeer de steunvectoren
    support_vector_indices = np.where(
        np.abs(labels - svm_classification(data_with_intercept, weights)) != 0)[0]
    ax.scatter(data[support_vector_indices, 0], data[support_vector_indices, 1], marker='o', facecolors='none',
               edgecolors='r', s=200, label='Steunvectoren')

    # Plaats de gegevenspunten in de plot
    scatter = ax.scatter(data[:, 0], data[:, 1],
                         c=labels, cmap=plt.cm.Paired, marker='o', s=50)

    ax.set_title(f'Epoch {frame + 1}/{animation_frames}')
    ax.legend()


# Aantal frames voor de animatie
animation_frames = 10

# Creëer de animatie
ani = animation.FuncAnimation(
    fig, update, frames=animation_frames, repeat=False)

# Toon de plot
plt.show()
