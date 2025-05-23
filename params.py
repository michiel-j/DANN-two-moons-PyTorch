"""
Params for DANN

Note that these parameters are not optimised well, as this repo is for demonstration purposes.
"""

num_epochs = 800 # Copied from https://adapt-python.github.io/adapt/examples/Two_moons.html
manual_seed = 562
manual_seed_weight_init = 123
batch_size = 34
learning_rate_no_dann = 1e-3 # Copied from https://adapt-python.github.io/adapt/examples/Two_moons.html#Source-Only
learning_rate_dann = 1e-3
beta1 = 0.5 # Copied from https://adapt-python.github.io/adapt/examples/Two_moons.html#Source-Only
beta2 = 0.999 # The default in both TensorFlow and PyTorch ADAM
weight_decay = 0
domain_adaptation_lambda = 1.0 # In range [0, 1], copied from https://adapt-python.github.io/adapt/examples/Two_moons.html#DANN
