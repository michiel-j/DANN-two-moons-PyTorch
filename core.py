import torch
from torch import nn
import numpy as np
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
            y_pred_batch = classifier(features_src)

            # Compute training loss
            loss = criterion(input=y_pred_batch.squeeze(), target=y_true_batch)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward(inputs=optim_params)
            optimizer.step()

            losses_within_epoch.extend([loss.item()] * len(y_true_batch)) # Let's assume len(y_true_batch) is equal to batch size. This allows for incomplete batches.
        
        losses_across_epochs.append(np.mean(np.array(losses_within_epoch)))
        
        # Print metrics for this epoch
        if epoch % 10 == 0:
            print(f'Epoch {epoch} of {params.num_epochs}: train loss is {losses_across_epochs[-1]:.6f}')

    # Plot training loss curve
    plt = utils.plot_loss_curve(losses_across_epochs)
    plt.savefig('./plots/losses/train_loss_no_domain_adaptation.png')
    plt.close()

    return encoder, classifier


def eval(encoder, classifier, data_loader, print_output: bool = True, is_discriminator: bool = False):
    """
    Evaluate encoder and classifier (or discriminator) using either source or target domain data.
    """
    encoder.eval()
    classifier.eval()
    
    predictions, true_labels = [], [] # Let's use list because it is easiest to append to

    criterion = nn.BCELoss()

    # Evaluate network
    with torch.no_grad():
        for idx, (X_batch, y_true_batch) in enumerate(data_loader):
            # Extract features using encoder network
            features = encoder(X_batch)

            # Predict source samples on classifier
            y_pred_batch = classifier(features).squeeze()
            predictions.extend(y_pred_batch.detach().cpu().tolist())
            true_labels.extend(y_true_batch.detach().cpu().tolist())

        # Compute training loss
        predictions = np.array(predictions, dtype=np.float32)
        true_labels = np.array(true_labels, dtype=np.float32)
        loss = criterion(
            input=torch.from_numpy(predictions),
            target=torch.from_numpy(true_labels)
        )
        thresholded_predictions = np.where(predictions >= 0.5, 1, 0) # TODO: .squeeze()

    if print_output:
        if is_discriminator:
            # In case the classifier argument is a discriminator (domain classifier)
            print(f"\t avg loss = {loss:.6f}, avg domain cl acc = {accuracy_score(y_true=true_labels, y_pred=thresholded_predictions):2%}")
        else:
            print(f"\t avg loss = {loss:.6f}, avg acc = {accuracy_score(y_true=true_labels, y_pred=thresholded_predictions):2%}, ARI = {adjusted_rand_score(labels_true=true_labels, labels_pred=thresholded_predictions):.4f}")
    return thresholded_predictions, true_labels


def train_src_tgt(encoder, classifier, discriminator, src_data_loader, tgt_data_loader):
    """
    Train encoder to learn domain-invariant features (DANN setup).
    
    In the ADAPT Two Moons example, a gamma factor is given for speedup. This gamma factor
    is used in the calculation of lambda (see ADAPT's UpdateLambda callback). I have copied
    the formula from the unsupervised DANN paper by Ganin et al.
    This gamma/lambda value can also be set to a constant. In this case, it is recommended to
    first train the encoder and classifier only using source data, then fine-tuning the entire
    DANN using both source and target data.

    Another approach to domain adaptation is to start from a pre-trained encoder and classifier,
    and using a fixed lambda. To do this, set lambda to the constant value (e.g. 0.05), and disable
    the calculating of lambda (lambda grows from 0 to 1).
    """
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
    criterion_classifier = nn.BCELoss()

    # for epoch in range(params.num_epochs):
    for epoch in range(params.num_epochs_classifier_discriminator_encoder_equal_norm):
        classifier_losses_within_epoch = []

        for step, ((X_batch_src, y_true_batch_src), (X_batch_tgt, _)) in enumerate(zip(src_data_loader, tgt_data_loader)):
            # Get the value for lambda, which will be used to weight the discriminator loss term in the encoder loss calculation
            # lambda_ = utils.get_lambda_value_domain_adaptation( # Use this lambda when starting with a randomly initialised encoder, classifier and discriminator.
            #     current_step=step,
            #     current_epoch=epoch,
            #     num_epochs=params.num_epochs,
            #     len_dataloader=len(src_data_loader)
            # ) # Use this lambda going from 0 to 1 for approach 1
            lambda_ = 0.045 # Use this lambda (0.042 or 0.045) for approach 2.

            # Source domain data
            features_src = encoder(X_batch_src) # Extract features using encoder network
            y_pred_batch = classifier(features_src) # Predict source samples on classifier
            classifier_loss = criterion_classifier(input=y_pred_batch.squeeze(), target=y_true_batch_src) # Compute classifier's training loss
            discriminator_pred_src = discriminator(features_src) # Domain predictions

            # Target domain data
            features_tgt = encoder(X_batch_tgt) # Extract features using encoder network
            discriminator_pred_tgt = discriminator(features_tgt) # Domain prediction

            # Compute discriminator loss
            discriminator_loss = torch.mean(-torch.log(discriminator_pred_src + torch.finfo(torch.float32).eps) - torch.log(1 - discriminator_pred_tgt + torch.finfo(torch.float32).eps)) # Compute discriminator's training loss

            # Compute encoder loss, assume lambda_ = 1.0 (ADAPT's value in two moons example)
            encoder_loss = classifier_loss - lambda_ * discriminator_loss

            # Backpropagation
            optimizer.zero_grad()
            classifier_loss.backward(inputs=list(classifier.parameters()), retain_graph=True)
            discriminator_loss.backward(inputs=list(discriminator.parameters()), retain_graph=True)
            encoder_loss.backward(inputs=list(encoder.parameters()), retain_graph=True)
            optimizer.step()

            classifier_losses_within_epoch.extend([classifier_loss.item()] * len(y_true_batch_src)) # Let's assume len(y_true_batch) is equal to batch size. This allows for incomplete batches.
        
        classifier_losses_across_epochs.append(np.mean(np.array(classifier_losses_within_epoch)))
        
        # Print metrics for this epoch
        if epoch % 10 == 0:
            # print(f'Epoch {epoch} of {params.num_epochs}: train loss is {classifier_losses_across_epochs[-1]:.6f} (lambda is {lambda_:.4f})')
            print(f'Epoch {epoch} of {params.num_epochs_classifier_discriminator_encoder_equal_norm}: train loss is {classifier_losses_across_epochs[-1]:.6f} (lambda is {lambda_:.4f})')

    # Plot training loss curve
    plt = utils.plot_loss_curve(classifier_losses_across_epochs)
    plt.savefig('./plots/losses/train_loss_with_domain_adaptation.png')
    plt.close()

    return encoder, classifier, discriminator


def train_src_tgt_equal_norm(encoder, classifier, discriminator, src_data_loader, tgt_data_loader):
    """
    Train encoder to learn domain-invariant features (DANN setup).
    
    In this implementation, the tuning of the lambda hyperparameter is eliminated. This is
    achieved by setting 
    """
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
    criterion_classifier = nn.BCELoss()

    # for epoch in range(params.num_epochs):
    for epoch in range(params.num_epochs_classifier_discriminator_encoder_equal_norm):
        classifier_losses_within_epoch = []

        # Source domain data
        for step, ((X_batch_src, y_true_batch_src), (X_batch_tgt, _)) in enumerate(zip(src_data_loader, tgt_data_loader)):
            # Get the value for lambda, which will be used to weight the discriminator loss term in the encoder loss calculation
            lambda_ = -1.5

            # Source domain data
            features_src = encoder(X_batch_src) # Extract features using encoder network
            y_pred_batch = classifier(features_src) # Predict source samples on classifier
            classifier_loss = criterion_classifier(input=y_pred_batch.squeeze(), target=y_true_batch_src) # Compute classifier's training loss
            discriminator_pred_src = discriminator(features_src) # Domain predictions

            # Target domain data
            features_tgt = encoder(X_batch_tgt) # Extract features using encoder network
            discriminator_pred_tgt = discriminator(features_tgt) # Domain prediction

            # Compute discriminator loss
            discriminator_pred = torch.cat((-torch.log(discriminator_pred_src + torch.finfo(torch.float32).eps), - torch.log(1 - discriminator_pred_tgt + torch.finfo(torch.float32).eps)), dim=0).squeeze()
            # discriminator_loss = torch.mean(-torch.log(discriminator_pred_src + torch.finfo(torch.float32).eps) - torch.log(1 - discriminator_pred_tgt + torch.finfo(torch.float32).eps)) # Compute discriminator's training loss
            discriminator_loss = torch.mean(discriminator_pred)

            # Compute gradient of classifier loss
            gradient_classifier_loss = torch.autograd.grad(classifier_loss, features_src, grad_outputs=torch.ones_like(classifier_loss), retain_graph=True)[0]
            norm_gradient_classifier_loss = torch.linalg.vector_norm(gradient_classifier_loss.reshape([gradient_classifier_loss.shape[0], -1]), ord=2, dim=-1)

            # Compute gradient of discriminator loss
            # TODO: why only derive to features_src and not to features_tgt?
            gradient_discriminator_loss_src = torch.autograd.grad(discriminator_loss, features_src, grad_outputs=torch.ones_like(discriminator_loss), retain_graph=True)[0]
            gradient_discriminator_loss_tgt = torch.autograd.grad(discriminator_loss, features_tgt, grad_outputs=torch.ones_like(discriminator_loss), retain_graph=True)[0]
            gradient_discriminator_loss = torch.cat((gradient_discriminator_loss_src, gradient_discriminator_loss_tgt), dim=0)
            norm_gradient_discriminator_loss = torch.linalg.vector_norm(gradient_discriminator_loss.reshape([gradient_discriminator_loss.shape[0], -1]), ord=2, dim=-1)
            
            imputation_value_classifier_grad_norm = norm_gradient_classifier_loss.max() # Max or mean. Empirically, max works better
            norm_gradient_classifier_loss = torch.cat((norm_gradient_classifier_loss, torch.zeros_like(norm_gradient_classifier_loss)), dim=0)
            norm_gradient_classifier_loss = torch.where(norm_gradient_classifier_loss == 0, imputation_value_classifier_grad_norm, norm_gradient_classifier_loss) # Max or mean

            # Compute encoder loss, assume lambda_ = 1.0 (ADAPT's value in two moons example)
            # encoder_loss_equal_norm = torch.mean(-torch.mean(torch.log(torch.nn.functional.sigmoid(discriminator_logits_source) + torch.finfo(torch.float32).eps), dim=-1) / (torch.maximum(norm_grad_meta_loss_source.detach(), torch.tensor(torch.finfo(torch.float32).eps))) * norm_grad_class_loss.detach() * 1.5) # + torch.log(1 - torch.nn.functional.sigmoid(discriminator_logits_source) + torch.tensor(torch.finfo(torch.float32).eps))
            # encoder_loss_equal_norm = torch.mean(torch.mean(-torch.log(discriminator_pred_src + torch.finfo(torch.float32).eps) - torch.log(1 - discriminator_pred_tgt + torch.finfo(torch.float32).eps)) / (torch.maximum(norm_gradient_discriminator_loss.detach(), torch.tensor(torch.finfo(torch.float32).eps))) * norm_gradient_classifier_loss.detach() * 1.5)
            
            # TODO: why divide one loss value by batch of max/eps values?
            discriminator_pred = torch.cat((-torch.log(discriminator_pred_src + torch.finfo(torch.float32).eps), - torch.log(1 - discriminator_pred_tgt + torch.finfo(torch.float32).eps)), dim=0).squeeze()
            encoder_loss_equal_norm = torch.mean(discriminator_pred / (torch.maximum(norm_gradient_discriminator_loss.detach(), torch.tensor(torch.finfo(torch.float32).eps))) * norm_gradient_classifier_loss.detach() * lambda_)

            encoder_loss_equal_norm += classifier_loss

            # Backpropagation: the backprop of classifier and encoder can be merged from two operations into one
            optimizer.zero_grad()
            classifier_loss.backward(inputs=list(classifier.parameters()), retain_graph=True)
            discriminator_loss.backward(inputs=list(discriminator.parameters()), retain_graph=True)
            encoder_loss_equal_norm.backward(inputs=list(encoder.parameters()), retain_graph=True)
            optimizer.step()

            classifier_losses_within_epoch.extend([classifier_loss.item()] * len(y_true_batch_src)) # Let's assume len(y_true_batch) is equal to batch size. This allows for incomplete batches.
        
        classifier_losses_across_epochs.append(np.mean(np.array(classifier_losses_within_epoch)))
        
        # Print metrics for this epoch
        if epoch % 10 == 0:
            print(f'Epoch {epoch} of {params.num_epochs_classifier_discriminator_encoder_equal_norm}: train loss is {classifier_losses_across_epochs[-1]:.6f} (equal norm)')

    # Plot training loss curve
    plt = utils.plot_loss_curve(classifier_losses_across_epochs)
    plt.savefig('./plots/losses/train_loss_with_domain_adaptation_equal_norm.png')
    plt.close()

    return encoder, classifier, discriminator


def train_discriminator(encoder, discriminator, src_data_loader, tgt_data_loader):
    """
    (Pre-)Train discriminator to distinguish between source and target domains.
    """
    losses_across_epochs = []

    encoder.eval()
    discriminator.train()

    # Setup criterion and optimizer
    optim_params = list(discriminator.parameters())
    optimizer = torch.optim.Adam(
        optim_params,
        lr=1e-2, # Hard-coded to not create an additional parameter
        betas=(params.beta1, params.beta2)
    )
    
    criterion = nn.BCELoss(reduction='none')

    for epoch in range(params.num_epochs_pretrain_discriminator_equal_norm):
        losses_within_epoch = []
        for step, ((X_batch_src, _), (X_batch_tgt, _)) in enumerate(zip(src_data_loader, tgt_data_loader)):
            # Source domain data
            features_src = encoder(X_batch_src) # Extract features using encoder network
            discriminator_pred_src = discriminator(features_src) # Predict source samples on classifier
            
            # Target domain data
            features_tgt = encoder(X_batch_tgt) # Extract features using encoder network
            discriminator_pred_tgt = discriminator(features_tgt) # Predict source samples on classifier

            # Compute discriminator loss
            discriminator_pred = torch.cat((-torch.log(discriminator_pred_src + torch.finfo(torch.float32).eps), - torch.log(1 - discriminator_pred_tgt + torch.finfo(torch.float32).eps)), dim=0).squeeze()
            discriminator_loss = torch.mean(discriminator_pred)

            # Backpropagation
            optimizer.zero_grad()
            discriminator_loss.backward(inputs=optim_params)
            optimizer.step()

            losses_within_epoch.extend([discriminator_loss.item()] * X_batch_src.shape[0]) # Let's assume X_batch_src.shape[0] is equal to batch size. This allows for incomplete batches.
        
        losses_across_epochs.append(np.mean(np.array(losses_within_epoch)))
        
        # Print metrics for this epoch
        if epoch % 10 == 0:
            print(f'Epoch {epoch} of {params.num_epochs_pretrain_discriminator_equal_norm}: train loss is {losses_across_epochs[-1]:.6f}')

    # Plot training loss curve
    plt = utils.plot_loss_curve(losses_across_epochs)
    plt.savefig('./plots/losses/train_loss_pretrain-discriminator_equal_norm.png')
    plt.close()

    return encoder, discriminator


