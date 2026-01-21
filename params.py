"""
Params for DANN

Note that these parameters are not optimised well, as this repo is for demonstration purposes.
"""

num_epochs = 800 # Copied from https://adapt-python.github.io/adapt/examples/Two_moons.html
num_epochs_pretrain_discriminator_equal_norm = 800 # Only train discriminator. Do not train encoder nor classifier
num_epochs_classifier_discriminator_encoder_equal_norm = 2000
manual_seed = 562
manual_seed_weight_init = 9 # I admit this seed was a bit cherry-picked to match the ADAPT example
batch_size = 34
learning_rate_no_dann = 1e-3 # Copied from https://adapt-python.github.io/adapt/examples/Two_moons.html#Source-Only
learning_rate_dann = 1e-3
beta1 = 0.5 # Copied from https://adapt-python.github.io/adapt/examples/Two_moons.html#Source-Only
beta2 = 0.999 # The default in both TensorFlow and PyTorch ADAM
weight_decay = 0
domain_adaptation_lambda = 1.0 # In range [0, 1], copied from https://adapt-python.github.io/adapt/examples/Two_moons.html#DANN
