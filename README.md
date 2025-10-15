<!--             
<style>
  .texttt {
    font-family: Consolas; /* Monospace font */
    font-size: 1em; /* Match surrounding text size */
    color: teal; /* Add this line to set text color to blue */
    letter-spacing: 0; /* Adjust if needed */
  }
</style> -->

<h1 align="center">
  <span style="color: teal; font-family: Consolas;">Learning Compact Representations of Agricultural Fields</span>: A Study of Variational Autoencoders Variants for Aerial Drone Imagery
</h1>


## 📝 Abstract
The integration of aerial drone imagery into precision agriculture enables large-scale, high-resolution monitoring of crop conditions but also introduces challenges due to the dimensionality and variability of visual data. Variational Autoencoders (VAEs) and their variants provide a promising framework for learning compact latent representations that preserve meaningful crop features while reducing computational complexity. In this work, we investigate the suitability of VAE-based architectures for analyzing aerial imagery of pineapple fields. Using a dataset of approximately 5000 drone-acquired images, we evaluate reconstruction fidelity, robustness to noise, and the structure of latent spaces across different VAE variants, including VQ-VAE and \beta-VAE. Our analysis demonstrates that VAEs not only capture fine-grained plant features but also enable latent-space clustering that separates foreground (pineapple plants) from background (soil, vegetation, and other elements). These findings highlight the potential of VAE-derived representations for supporting downstream tasks such as plant counting, yield estimation, and stress detection. By releasing both dataset and code, we aim to establish a reproducible benchmark for generative models in agricultural monitoring, advancing data-driven approaches for sustainable pineapple cultivation.



This repository provides implementations of Autoencoders and Variational Autoencoder (VAE) variants designed for learning compact, informative representations from aerial drone imagery of agricultural fields. These models are intended to support key tasks in **precision agriculture**, including **crop monitoring**, **yield estimation**, and **field analysis**.

## Installation 
We tested our code using CUDA 12.8 and Python 3.10. To install requirements, run in the  terminal:
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

