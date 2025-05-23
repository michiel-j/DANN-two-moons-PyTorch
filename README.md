# DANN-two-moons-PyTorch
Demo of an unsupervised domain adversarial neural network (DANN) using the Two Moons synthetic dataset. Implemented in PyTorch, compared to [ADAPT package Two Moons](https://adapt-python.github.io/adapt/examples/Two_moons.html) example to ensure correctness. \
The hyperparameters were copied from the ADAPT demo. Therefore, these may be suboptimal.

## Dataset
To ensure identical data as used in the [ADAPT Package Two Moons](https://adapt-python.github.io/adapt/examples/Two_moons.html) example, I have copied the data creation function (`make_moons_da()` in `utils.py`) to ensure the same data was used. \
Additionally, I have added an independent test set, on which no model weights were fitted. The code to create the dataset can be found in `utils.py`.

Visualisation of the Two Moons training data:
<p align="center">
  <img width="500" src=./plots/train_dataset_samples.png>
</p>


## Model
The encoder and classifier were copied from the [ADAPT example](https://adapt-python.github.io/adapt/examples/Two_moons.html#Network). The discriminator is ADAPT's default and was copied from the [ADAPT DANN docs](https://adapt-python.github.io/adapt/generated/adapt.feature_based.DANN.html). The definitions of encoder, classifier and discriminator can be found in `models.py`.

## Results
### No domain adaptation
First, only the encoder and the classifier are trained. No domain adaptation is applied.

Visualisation of the network without domain adaptation:
<p align="center">
  <img width="500" src=./plots/contour_plot_no_domain_adaptation.png>
</p>

The corresponding results on the test set data are (`log.txt`):
```
Source domain test data: avg loss = 0.000000, avg acc = 100.000000%, ARI = 1.0000
Target domain test data: avg loss = 18.750000, avg acc = 85.000000%, ARI = 0.4849
```

Visualisation of the feature extractor's latent space (encoder) without domain adaptation (through PCA):
<p align="center">
  <img width="500" src=./plots/pca_encoder_no-domain-adaptation.png>
</p>

### With domain adaptation
Next, the encoder, the classifier and the discriminator (domain classifier) were trained. Domain adaptation was applied.

Visualisation of the network with domain adaptation:
<p align="center">
  <img width="500" src=./plots/contour_plot_with_domain_adaptation.png>
</p>

The corresponding results on the test set data are (`log.txt`):
```
Source domain test data: avg loss = 0.000000, avg acc = 98.000000%, ARI = 0.9208
Target domain test data: avg loss = 12.500000, avg acc = 93.000000%, ARI = 0.7370
```
The hyperparameters were copied from the ADAPT demo. Therefore, these may be suboptimal.

Visualisation of the feature extractor's latent space (encoder) with domain adaptation (through PCA):
<p align="center">
  <img width="500" src=./plots/pca_encoder_with-domain-adaptation.png>
</p>

## Reproducibility
The minimal list of required packages can be found in `requirements.txt`. All code was tested in Python 3.11 and 3.12 with PyTorch 2.6.0 and 2.4.1 resp. \
Random seeds are set to ensure reproducibility. These can be found in `params.py`.

## References
This repository is mainly based on the ADAPT package (TensorFlow only), and some ideas (plotting, file structure) originated when looking at mashaan14's DANN toy example:
- *Awesome Domain Adaptation in Python Toolbox* (ADAPT): [https://adapt-python.github.io/adapt/index.html](https://adapt-python.github.io/adapt/index.html)
- *DANN-toy* repository from mashaan14: [https://github.com/mashaan14/DANN-toy](https://github.com/mashaan14/DANN-toy)

More information about the unsupervised DANN can be found in the following article:
```bibtex
@article{JMLR:v17:15-239,
  author  = {Yaroslav Ganin and Evgeniya Ustinova and Hana Ajakan and Pascal Germain and Hugo Larochelle and Fran{\c{c}}ois Laviolette and Mario March and Victor Lempitsky},
  title   = {Domain-Adversarial Training of Neural Networks},
  journal = {Journal of Machine Learning Research},
  year    = {2016},
  volume  = {17},
  number  = {59},
  pages   = {1--35},
  url     = {http://jmlr.org/papers/v17/15-239.html}
}
```
