from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
import numpy as np


def plot_dataset(Xs, Xt, ys):
    """
    Copied from https://adapt-python.github.io/adapt/examples/Two_moons.html#Setup
    """
    plt.figure(figsize=(6, 5))
    plt.title("Input space")
    plt.scatter(Xs[ys==0, 0], Xs[ys==0, 1], label="source, 0", edgecolors='k', c="red") # Source domain label 0: red colour
    plt.scatter(Xs[ys==1, 0], Xs[ys==1, 1], label="source, 1", edgecolors='k', c="blue") # Source domain label 1: blue colour
    plt.scatter(Xt[:, 0], Xt[:, 1], label="target", edgecolors='k', c="black") # Target domain: black colour
    plt.legend(loc="lower right")
    plt.tight_layout()
    return plt


def plot_loss_curve(train_losses: list):
    plt.figure()
    plt.plot(train_losses, linewidth=1)
    plt.xlabel('Epochs')
    plt.ylabel('Train loss value')
    plt.title('Training loss history')
    plt.tight_layout()
    return plt


def plot_contour(Xs, Xt, ys, x_grid, y_grid, y_pred_grid):
    plt.figure()
    plt.title("Contour plot of input space")
    plt.contourf(x_grid, y_grid, y_pred_grid, cmap=plt.cm.RdBu, alpha=0.6)
    plt.scatter(Xs[ys==0, 0], Xs[ys==0, 1], label="source, 0", edgecolors='k', c="red")
    plt.scatter(Xs[ys==1, 0], Xs[ys==1, 1], label="source, 1", edgecolors='k', c="blue")
    plt.scatter(Xt[:, 0], Xt[:, 1], label="target", edgecolors='k', c="black")
    plt.legend()
    plt.tight_layout()
    return plt


def make_moons_da(n_samples=100, rotation=30, noise=0.05, random_state=0):
    """
    Function to make source and target domain data from sklearn.datasets.make_moons.
    Copied from https://adapt-python.github.io/adapt/examples/Two_moons.html#Setup
    """
    Xs, ys = make_moons(n_samples=n_samples,
                        noise=noise,
                        random_state=random_state)
    Xs[:, 0] -= 0.5
    theta = np.radians(-rotation)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rot_matrix = np.array(
        ((cos_theta, -sin_theta),
         (sin_theta, cos_theta))
    )
    Xt = Xs.dot(rot_matrix)
    yt = ys
    return Xs, ys, Xt, yt
