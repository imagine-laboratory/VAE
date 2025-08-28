# Variational Autoencoders for Aerial Agricultural Analysis

This repository provides implementations of Autoencoders and Variational Autoencoder (VAE) variants designed for learning compact, informative representations from aerial drone imagery of agricultural fields. These models are intended to support key tasks in **precision agriculture**, including **crop monitoring**, **yield estimation**, and **field analysis**.

## 🚀 Getting Started
We tested our code using CUDA 12.8. To install requirements, run in the  terminal:
```bash
pip3 install -r requirements.txt
```

## 🔍 Overview of Models

- **Autoencoder (AE)**  
  A standard encoder–decoder model for unsupervised feature extraction and image reconstruction.

- **Variational Autoencoder (VAE)**  
  A probabilistic generative model that learns a latent distribution over the input space, enabling stochastic sampling and smooth interpolation.

- **β-VAE**  
  A variant of VAE that introduces a hyperparameter β to promote disentangled latent representations, useful for more interpretable factors of variation.

- **Vector-Quantized VAE (VQ-VAE)**  
  A discrete latent variable model that uses vector quantization to learn a finite set of latent embeddings, improving interpretability and compression.

## 🚀 Getting Started

To train or evaluate a model, use the following command:

```bash
python main.py --config ./configs/vae_perceptual.yaml --model vqvae
```

