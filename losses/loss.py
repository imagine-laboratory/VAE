import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from typing import Optional, List, Dict

def convnext_perceptual_loss(    
    x_real,
    x_recon,
    convnext_loss
):
    loss_value = convnext_loss(x_recon, x_real)
    return loss_value
    
    

def dino_perceptual_loss(
    x_real,
    x_recon,
    dino_model,
    layer_ids=[11],
    mode='cls',         # 'cls', 'mean', or 'tokens'
    reduction='mean'    # or 'none'
):
    """
    Compute perceptual loss between x_real and x_recon using DINO ViT features.

    Args:
        x_real (Tensor): Original image batch [B, 3, H, W]
        x_recon (Tensor): Reconstructed image batch [B, 3, H, W]
        dino_model (nn.Module): DINO ViT model with get_intermediate_layers
        layer_ids (list[int]): Layer indices to use for perceptual comparison
        mode (str): 'cls' | 'mean' | 'tokens'
        reduction (str): 'mean' | 'sum' | 'none'

    Returns:
        Tensor: Scalar loss (or per-sample if reduction='none')
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=x_real.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x_real.device).view(1, 3, 1, 1)

    x_real = (x_real - mean) / std
    x_recon = (x_recon - mean) / std

    # Get intermediate layers
    feats_recon = dino_model.get_intermediate_layers(x_recon, n=len(dino_model.blocks)+1)

    with torch.no_grad():
        feats_real = dino_model.get_intermediate_layers(x_real, n=len(dino_model.blocks)+1)

    loss = 0.0

    for layer in layer_ids:
        f_real = feats_real[layer]  # [B, T, D]
        f_recon = feats_recon[layer]

        if mode == 'cls':
            v_real = f_real[:, 0]    # CLS token
            v_recon = f_recon[:, 0]

        elif mode == 'mean':
            v_real = f_real.mean(dim=1)
            v_recon = f_recon.mean(dim=1)

        elif mode == 'tokens':
            v_real = f_real
            v_recon = f_recon

        else:
            raise ValueError(f"Unknown mode: {mode}")

        loss += F.mse_loss(v_real, v_recon, reduction=reduction)

    return loss #/ len(layer_ids)

def mse_loss(reconstructed, original, reduction='sum'):
    """
    Computes MSE reconstruction loss.

    Args:
        reconstructed: Reconstructed image tensor [B, C, H, W]
        original: Original image tensor [B, C, H, W]
        reduction: 'sum' or 'mean' for loss aggregation

    Returns:
        MSE loss (float)
    """
    loss_fn = nn.MSELoss(reduction=reduction)
    return loss_fn(reconstructed, original)


def kl_divergence_loss(mean, logvar, reduction='sum'):
    """
    Computes the KL divergence between the latent distribution and standard normal.

    Args:
        mean: Mean of latent distribution [B, latent_dim]
        logvar: Log variance of latent distribution [B, latent_dim]
        reduction: 'sum' or 'mean' for loss aggregation

    Returns:
        KL divergence loss (float)
    """
    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    if reduction == 'mean':
        return kl / mean.size(0)
    return kl




def vae_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    kl_beta: float = 0.1,
    reduction: str = 'sum',
) -> Dict[str, torch.Tensor]:
    """
    Computes the VAE loss: MSE + β * KL divergence + optional perceptual loss.

    Args:
        reconstructed (Tensor): Reconstructed images [B, C, H, W]
        original (Tensor): Original images [B, C, H, W]
        mean (Tensor): Latent means [B, latent_dim]
        logvar (Tensor): Latent log-variances [B, latent_dim]
        kl_beta (float): Scaling factor for KL divergence.
        reduction (str): Reduction method: 'sum' or 'mean'.
        perceptual_loss (bool): Whether to include perceptual loss.
        model_perceptual (nn.Module, optional): DINO model for perceptual loss.
        layers_ids (List[int], optional): List of layer indices.
        mode (str): Feature mode: 'cls', 'mean', or 'tokens'.

    Returns:
        Dict[str, Tensor]: Dictionary with keys:
            - 'total'
            - 'reconstruction'
            - 'kl'
            - 'perceptual' (only if enabled)
    """
    batch_size = reconstructed.size(0)

    # Core losses
    recon_loss = mse_loss(reconstructed, original, reduction=reduction)
    kl_loss = kl_divergence_loss(mean, logvar, reduction=reduction)
    total_loss = recon_loss + kl_beta * kl_loss

    # Normalize all by batch size
    loss_dict = {
        "total": total_loss / batch_size,
        "reconstruction": recon_loss / batch_size,
        "kl": kl_loss / batch_size,
    }

    return loss_dict

def vae_perceptual_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    kl_beta: float = 0.1,
    reduction: str = 'sum',
    perceptual_loss: bool = False,
    model_perceptual: Optional[torch.nn.Module] = None,
    layers_ids: Optional[List[int]] = None,
    mode: str = 'cls', model_name: str= "dino"
) -> Dict[str, torch.Tensor]:
    """
    Computes the VAE loss: MSE + β * KL divergence + optional perceptual loss.

    Args:
        reconstructed (Tensor): Reconstructed images [B, C, H, W]
        original (Tensor): Original images [B, C, H, W]
        mean (Tensor): Latent means [B, latent_dim]
        logvar (Tensor): Latent log-variances [B, latent_dim]
        kl_beta (float): Scaling factor for KL divergence.
        reduction (str): Reduction method: 'sum' or 'mean'.
        perceptual_loss (bool): Whether to include perceptual loss.
        model_perceptual (nn.Module, optional): DINO model for perceptual loss.
        layers_ids (List[int], optional): List of layer indices.
        mode (str): Feature mode: 'cls', 'mean', or 'tokens'.

    Returns:
        Dict[str, Tensor]: Dictionary with keys:
            - 'total'
            - 'reconstruction'
            - 'kl'
            - 'perceptual' (only if enabled)
    """
    batch_size = reconstructed.size(0)

    # Core losses
    recon_loss = mse_loss(reconstructed, original, reduction=reduction)
    kl_loss = kl_divergence_loss(mean, logvar, reduction=reduction)
    total_loss = recon_loss + kl_beta * kl_loss

    # Optional perceptual loss
    perceptual = None
    if perceptual_loss:
        if model_perceptual is None or layers_ids is None:
            raise ValueError("Perceptual loss is enabled but model_perceptual or layers_ids is not provided.")
        if model_name == "dino" or model_name == "dinov2":
            perceptual = dino_perceptual_loss(
                original,
                reconstructed,
                model_perceptual,
                layer_ids=layers_ids,
                mode=mode,
                reduction=reduction
            )
            total_loss += perceptual
        elif model_name == "convnext":
            perceptual = convnext_perceptual_loss(
                original,
                reconstructed,
                model_perceptual
            )
            total_loss += perceptual

    # Normalize all by batch size
    loss_dict = {
        "total": total_loss / batch_size,
        "reconstruction": recon_loss / batch_size,
        "kl": kl_loss / batch_size,
    }

    if perceptual is not None:
        loss_dict["perceptual"] = perceptual / batch_size

    return loss_dict


def vqvae_loss(recon_x, x, vq_loss):
    #recon_loss = F.mse_loss(recon_x, x)
    b_size = recon_x.size(0)
    loss_fn = nn.MSELoss(reduction="sum")
    recon_loss = loss_fn(recon_x, x)
    total_loss = recon_loss + vq_loss

    return total_loss/b_size, recon_loss/b_size, vq_loss/b_size