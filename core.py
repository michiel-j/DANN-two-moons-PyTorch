import torch
from torch import nn
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score

import params
import utils

def train_src(encoder, classifier, src_data_loader):
    """
    Train encoder and classifier using source domain data
    """
    losses_across_epochs = []

    encoder.train()
    classifier.train()

    # Setup criterion and optimizer
    optim_params = list(classifier.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(
        optim_params,
        lr=params.learning_rate_no_dann,
        betas=(params.beta1, params.beta2)
    )
    
    criterion = nn.BCELoss()

    for epoch in range(params.num_epochs):
        losses_within_epoch = []
        for step, (X_batch, y_true_batch) in enumerate(src_data_loader):
            # Extract features using encoder network
            features_src = encoder(X_batch)

            # Predict source samples on classifier
            y_pred_batch = classifier(features_src) # y_pred_batch_logits

            # Compute training loss
            loss = criterion(input=y_pred_batch.squeeze(), target=y_true_batch)
            losses_within_epoch.append(loss.item())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward(inputs=optim_params)
            optimizer.step()
        
        losses_across_epochs.append(np.mean(np.array(losses_within_epoch)))
        
        # Print metrics for this epoch
        if epoch % 10 == 0:
            print(f'Epoch {epoch} of {params.num_epochs}: train loss is {losses_across_epochs[-1]:.6f}')

    # Plot training loss curve
    plt = utils.plot_loss_curve(losses_across_epochs)
    plt.savefig('./plots/losses/train_loss_no_domain_adaptation.png')
    plt.close()

    return encoder, classifier


def eval(encoder, classifier, src_data_loader, print_output: bool = True):
    """
    Evaluate encoder and classifier using source domain data
    """
    losses, true_labels, predicted_labels = [], [], []

    encoder.eval()
    classifier.eval()

    criterion = nn.BCELoss()

    # Evaluate network
    for idx, (X_batch, y_true_batch) in enumerate(src_data_loader):
        # Extract features using encoder network
        features_src = encoder(X_batch)

        # Predict source samples on classifier
        y_pred_batch = classifier(features_src).squeeze() # y_pred_batch_logits
        y_pred_batch = torch.where(y_pred_batch >= 0.5, 1, 0).to(dtype=torch.float32)

        # Compute training loss
        loss = criterion(input=y_pred_batch, target=y_true_batch)
        losses.append(loss.item())

        true_labels.extend(y_true_batch.tolist())
        predicted_labels.extend(y_pred_batch.tolist())

    if print_output:
        print(f"\t avg loss = {loss:.6f}, avg acc = {accuracy_score(y_true=true_labels, y_pred=predicted_labels):2%}, ARI = {adjusted_rand_score(labels_true=true_labels, labels_pred=predicted_labels):.4f}")
    return predicted_labels, true_labels


def train_src_tgt(encoder, classifier, discriminator, src_data_loader, tgt_data_loader):
    """
    Train encoder and classifier using source domain data
    """
    # Assume lambda_ 1.0 (ADAPT's default)
    lambda_ = params.domain_adaptation_lambda

    classifier_losses_across_epochs = []

    encoder.train()
    classifier.train()
    discriminator.train()

    # Setup criterion and optimizer
    optim_params = list(encoder.parameters()) + list(classifier.parameters()) + list(discriminator.parameters())
    optimizer = torch.optim.Adam(
        optim_params,
        lr=params.learning_rate_dann,
        betas=(params.beta1, params.beta2),
        weight_decay=params.weight_decay
    )
    criterion_classifier = nn.BCEWithLogitsLoss()

    for epoch in range(params.num_epochs):
        classifier_losses_within_epoch = []

        # Source domain data
        # for idx, (X_batch, y_true_batch) in enumerate(src_data_loader):
        for step, ((X_batch_src, y_true_batch_src), (X_batch_tgt, _)) in enumerate(zip(src_data_loader, tgt_data_loader)):
            # Source domain data
            features_src = encoder(X_batch_src) # Extract features using encoder network
            y_pred_batch = classifier(features_src) # Predict source samples on classifier
            classifier_loss = criterion_classifier(input=y_pred_batch.squeeze(), target=y_true_batch_src) # Compute classifier's training loss
            classifier_losses_within_epoch.append(classifier_loss.item())
            discriminator_logits_src = discriminator(features_src)

            # Target domain data
            features_tgt = encoder(X_batch_tgt) # Extract features using encoder network
            discriminator_logits_tgt = discriminator(features_tgt)

            # Compute discriminator loss
            discriminator_loss = torch.mean(-torch.log(torch.nn.functional.sigmoid(discriminator_logits_src) + torch.finfo(torch.float32).eps) - torch.log(1 - torch.nn.functional.sigmoid(discriminator_logits_tgt) + torch.finfo(torch.float32).eps)) # Compute discriminator's training loss

            # Compute encoder loss, assume lambda_ = 0.1 (ADAPT's default)
            encoder_loss = classifier_loss - lambda_ * discriminator_loss

            # Backpropagation
            optimizer.zero_grad()
            classifier_loss.backward(inputs=list(classifier.parameters()), retain_graph=True)
            discriminator_loss.backward(inputs=list(discriminator.parameters()), retain_graph=True)
            encoder_loss.backward(inputs=list(encoder.parameters()), retain_graph=True)
            optimizer.step()
        
        classifier_losses_across_epochs.append(np.mean(np.array(classifier_losses_within_epoch)))
        
        # Print metrics for this epoch
        if epoch % 10 == 0:
            print(f'Epoch {epoch} of {params.num_epochs}: train loss is {classifier_losses_across_epochs[-1]:.6f}')

    # Plot training loss curve
    plt = utils.plot_loss_curve(classifier_losses_across_epochs)
    plt.savefig('./plots/losses/train_loss_with_domain_adaptation.png')
    plt.close()

    return encoder, classifier, discriminator
