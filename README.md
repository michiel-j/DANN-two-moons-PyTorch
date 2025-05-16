# DANN-two-moons-PyTorch
Demo for unsupervised domain adversarial neural network (DANN) using Two Moons synthetic dataset. Implemented in PyTorch, compared to [ADAPT Package Two Moons](https://adapt-python.github.io/adapt/examples/Two_moons.html) example for correctness. \
The hyperparameters were copied from the ADAPT demo. Therefore, they may be suboptimal.

## Dataset
To be as close to the [ADAPT Package Two Moons](https://adapt-python.github.io/adapt/examples/Two_moons.html) example, I have copied the data creation to ensure the same data was used.
Additionally, I have added an independent test set, on which no model weights were fitted. The code to create the dataset can be found in `utils.py`.

Visualisation of the Two Moons training data:
<p align="center">
  <img width="500" src=./plots/train_dataset_samples.png>
</p>


## Model
The encoder and classifier were copied from [ADAPT demo (GitHub)](https://adapt-python.github.io/adapt/examples/Two_moons.html#Network). The discriminator is copied from the default as mentioned in [ADAPT DANN docs](https://adapt-python.github.io/adapt/generated/adapt.feature_based.DANN.html). The definitions of encoder, classifier and discriminator can be found in `models.py`.

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
Target domain test data: avg loss = 25.000000, avg acc = 88.000000%, ARI = 0.5734
```

### With domain adaptation
Next, the encoder, the classifier and the discriminator (domain classifier) are trained. Domain adaptation is applied.
Visualisation of the network with domain adaptation:
<p align="center">
  <img width="500" src=./plots/contour_plot_with_domain_adaptation.png>
</p>

The corresponding results on the test set data are (`log.txt`):
```
Source domain test data: avg loss = 0.000000, avg acc = 91.000000%, ARI = 0.6691
Target domain test data: avg loss = 0.000000, avg acc = 96.000000%, ARI = 0.8449
```
The hyperparameters were copied from the ADAPT demo. Therefore, they may be suboptimal.

## Reproducibility
The minimal list of required packages can be found in `requirements.txt`. All code was tested in Python 3.12 with PyTorch 2.4.1. \
Random seeds are set to ensure reproducibility. These can be found in `params.py`.

## References
I have mainly based this repository on the ADAPT package (TensorFlow only), and have drawn some inspiration (plotting, file structure) from mashaan14's DANN toy example:
- *Awesome Domain Adaptation in TensorFlow* (ADAPT): [https://adapt-python.github.io/adapt/index.html](https://adapt-python.github.io/adapt/index.html)
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
