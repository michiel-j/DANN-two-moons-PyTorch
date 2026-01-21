from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
import numpy as np
import torch


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


def plot_pca(encoder, src_data_loader, tgt_data_loader, file: str):
    """
    
    """
    encoder.eval()
    features_src_list, features_tgt_list = [], []
    labels_src_list = []

    with torch.no_grad():
        for idx, (X_batch, y_batch) in enumerate(src_data_loader):
            # Extract features using encoder network
            features_src = encoder(X_batch)
            features_src_list.append(features_src.detach().cpu())
            labels_src_list.append(y_batch.detach().cpu())

        for idx, (X_batch, _) in enumerate(tgt_data_loader):
            # Extract features using encoder network
            features_tgt = encoder(X_batch)
            features_tgt_list.append(features_tgt.detach().cpu())
    
    # Convert features from list to torch.Tensor by stacking
    features_src = torch.cat(features_src_list, dim=0).numpy()
    features_tgt = torch.cat(features_tgt_list, dim=0).numpy()
    labels_src = torch.cat(labels_src_list, dim=0).numpy()

    # Fit and transform the features using PCA
    pca_transformer = PCA(n_components=2).fit(np.concatenate((features_src, features_tgt), axis=0))
    features_src = pca_transformer.transform(features_src)
    features_tgt = pca_transformer.transform(features_tgt)

    # Plot the PCA
    plt.figure()
    plt.scatter(features_src[labels_src==0][:, 0], features_src[labels_src==0][:, 1], color='blue', s=3)
    plt.scatter(features_src[labels_src==1][:, 0], features_src[labels_src==1][:, 1], color='orange', s=3)
    plt.scatter(features_tgt[:, 0], features_tgt[:, 1], color='black', s=3)
    plt.xlabel('Latent dim. 1')
    plt.ylabel('Latent dim. 2')
    plt.title(f"PCA of encoder's latent space. \n Expl. variance is {(pca_transformer.explained_variance_ratio_[0]*100):.2f}% (dim. 1) and {(pca_transformer.explained_variance_ratio_[1]*100):.2f}% (dim. 2).")
    plt.legend(['source, 0', 'source, 1', 'target'])
    plt.tight_layout()
    plt.savefig(file)
    plt.close()
    return


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


def get_lambda_value_domain_adaptation(current_step: int, current_epoch: int, num_epochs: int, len_dataloader: int):
    """
    Calculate the lambda factor for domain adaptation, which evolves from 0 (beginning of training) to 1 (end of training).
    More information can be found in the unsupervised DANN paper by Ganin et al.
    """
    p = float(current_step + current_epoch * len_dataloader) / num_epochs / len_dataloader
    lambda_ = 2. / (1. + np.exp(-10 * p)) - 1
    return lambda_
