import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import random

import models
import utils
import core
import params


"""
Demo copied from ADAPT package (TensorFlow only): https://adapt-python.github.io/adapt/examples/Two_moons.html

Partly inspired by https://github.com/mashaan14/DANN-toy (file structure, plotting)
"""

def main():
    """
    In a first phase, train encoder and classifier.

    In a second phase, train encoder and classifier from their respective state.
    Use unsupervised DANN with a discriminator initialised from scratch. 
    """
    # Set random seeds for reproducibility (disabled CUDA settings since all is done on CPU)
    print(f"Random seed is {params.manual_seed}")
    random.seed(params.manual_seed)
    np.random.seed(params.manual_seed)
    torch.manual_seed(params.manual_seed)
    torch.use_deterministic_algorithms(True)
    # torch.cuda.manual_seed(params.manual_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    # Create source and target domain data
    # Assume 100 train samples and 100 test samples
    Xs, ys, Xt, yt = utils.make_moons_da(n_samples=200)

    # Create some variables for contour plots
    x_min, y_min = np.min([Xs.min(0), Xt.min(0)], 0)
    x_max, y_max = np.max([Xs.max(0), Xt.max(0)], 0)
    x_coord_grid, y_coord_grid = np.meshgrid(
        np.linspace(x_min-0.1, x_max+0.1, 100), np.linspace(y_min-0.1, y_max+0.1, 100)
    )
    X_grid = np.stack([x_coord_grid.ravel(), y_coord_grid.ravel()], -1)

    # Plot dataset with source (red and blue) and target (black) domain data
    plt = utils.plot_dataset(Xs[:100], Xt[:100], ys[:100])
    plt.savefig('./plots/train_dataset_samples.png')
    plt = utils.plot_dataset(Xs[100:], Xt[100:], ys[100:])
    plt.savefig('./plots/test_dataset_samples.png')

    # Create and initialise the encoder network and the classifier network (no discriminator yet)
    encoder = models.Encoder()
    classifier = models.Classifier()
    encoder.apply(models.init_weights)
    classifier.apply(models.init_weights)

    # Convert all data to torch.Tensor, such that device and dtype can be set. Next, create dataloaders for batching, shuffling, etc.
    Xs = torch.from_numpy(Xs).to(device='cpu', dtype=torch.float32)
    ys = torch.from_numpy(ys).to(device='cpu', dtype=torch.float32)
    Xt = torch.from_numpy(Xt).to(device='cpu', dtype=torch.float32)
    yt = torch.from_numpy(yt).to(device='cpu', dtype=torch.float32)
    X_grid = torch.from_numpy(X_grid).to(device='cpu', dtype=torch.float32)

    # Divide into train and test sets (100 samples each)
    Xs_train, Xs_test = Xs[:100, :], Xs[100:, :]
    ys_train, ys_test = ys[:100], ys[100:]
    Xt_train, Xt_test = Xt[:100, :], Xt[100:, :]
    yt_train, yt_test = yt[:100], yt[100:]

    Xs_loader_train = DataLoader(TensorDataset(Xs_train, ys_train), batch_size=params.batch_size, shuffle=True)
    Xt_loader_train = DataLoader(TensorDataset(Xt_train, yt_train), batch_size=params.batch_size, shuffle=True)
    Xs_loader_test = DataLoader(TensorDataset(Xs_test, ys_test), batch_size=params.batch_size, shuffle=False)
    Xt_loader_test = DataLoader(TensorDataset(Xt_test, yt_test), batch_size=params.batch_size, shuffle=False)
    X_loader_grid = DataLoader(TensorDataset(X_grid, torch.ones(X_grid.shape[0])), batch_size=params.batch_size, shuffle=False) # For contour plot. Ignore the labels (dummy var of -100), as these are necessary to iterate in core.eval function


    #########################################################################################################
    #                     Train encoder and classifier using source domain data only
    #########################################################################################################
    print(f'Train encoder and classifier using source domain data only for {params.num_epochs} epochs')
    encoder, classifier = core.train_src(encoder, classifier, Xs_loader_train)

    # Evaluate encoder and classifier using source domain data and target domain data
    print("Evaluation using src data (trained with only src data):")
    test_predictions_Xs_no_da, test_true_Xs_no_da = core.eval(encoder, classifier, Xs_loader_test)
    print("Evaluation using tgt data (trained with only src data):")
    test_predictions_Xt_no_da, test_true_Xt_no_da = core.eval(encoder, classifier, Xt_loader_test)

    # Create contour plot
    y_pred_grid, _ = core.eval(encoder, classifier, X_loader_grid, print_output=False) # print=False because this will give gibberish output
    y_pred_grid = np.array(y_pred_grid).reshape(100, 100) # Reshape back from separate 2 coordinate points to mesh/grid
    plt = utils.plot_contour(Xs, Xt, ys, x_coord_grid, y_coord_grid, y_pred_grid)
    plt.savefig('./plots/contour_plot_no_domain_adaptation.png')

    # Plot encoder's latent space through PCA
    core.pca_plots(encoder, Xs_loader_train, Xt_loader_train, './plots/pca_encoder_no-domain-adaptation.png')

    #########################################################################################################
    #  Train encoder, classifier and discriminator using source and target domain data (unsupervised DANN)
    #########################################################################################################
    # Create and initialise discriminator (domain classifier)
    encoder = models.Encoder()
    classifier = models.Classifier()
    encoder.apply(models.init_weights)
    classifier.apply(models.init_weights)
    discriminator = models.Discriminator()
    discriminator.apply(models.init_weights)
    
    print(f'Train encoder and classifier using unsupervised DANN for {params.num_epochs} epochs')
    encoder, classifier, discriminator = core.train_src_tgt(encoder, classifier, discriminator, Xs_loader_train, Xt_loader_train)

    # Evaluate encoder and classifier
    print("Evaluation using src data (trained with both src and tgt data):")
    test_predictions_Xs_with_da, test_true_Xs_with_da = core.eval(encoder, classifier, Xs_loader_test)
    print("Evaluation using tgt data (trained with both src and tgt data):")
    test_predictions_Xt_with_da, test_true_Xt_with_da = core.eval(encoder, classifier, Xt_loader_test)

    # Create contour plot
    y_pred_grid, _ = core.eval(encoder, classifier, X_loader_grid, print_output=False) # print=False because this will give gibberish output
    y_pred_grid = np.array(y_pred_grid).reshape(100, 100) # Reshape back from separate 2 coordinate points to mesh/grid
    plt = utils.plot_contour(Xs, Xt, ys, x_coord_grid, y_coord_grid, y_pred_grid)
    plt.savefig('./plots/contour_plot_with_domain_adaptation.png')

    # Plot encoder's latent space through PCA
    core.pca_plots(encoder, Xs_loader_train, Xt_loader_train, './plots/pca_encoder_with-domain-adaptation.png')
    
    return


main()
