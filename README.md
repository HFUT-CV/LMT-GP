# LMT-GP: Combined Latent Mean-Teacher and Gaussian Process for Semi-supervised Low-light Image Enhancement

# Introduction
This is a Pytorch implement of “LMT-GP: Combined Latent Mean-Teacher and Gaussian Process for Semi-supervised Low-light Image Enhancement” (ECCV 2024)

we propose a semi-supervised method based on latent mean-teacher and Gaussian process, named LMT-GP. We first design a latent mean-teacher framework that integrates both labeled and unlabeled data, as well as their latent vectors, into model training. Meanwhile, we use a mean-teacher-assisted Gaussian process learning strategy to establish a connection between the latent and pseudo-latent vectors obtained from the labeled and unlabeled data. To guide the learning process, we utilize an assisted Gaussian process regression (GPR) loss function.  Furthermore, we design a pseudo-label adaptation module (PAM) to ensure the reliability of the network learning.

# Installation

## Installing Package
Clone this repository to local disk and install:
```
pip install -r requirements
```
