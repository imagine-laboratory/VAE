import torch
from torch import nn
from torch.nn import functional as F
from typing import Literal, Optional, Tuple

from .residual import ResidualBlock
from .attention import AttentionBlock


class VAE_Encoder(nn.Module):
    """
    Drop-in replacement for your previous nn.Sequential-based encoder, with one addition:
    - forward(..., return_features: bool = False) -> when True, also returns the last spatial
      feature map `feats` (B x 512 x H/8 x W/8) right before projection to mean/logvar.
    
    The rest of the behavior (asymmetric padding on stride-2 layers, mean/logvar split, clamp,
    reparameterization with provided noise, and the 0.18215 scale) is identical.
    """

    def __init__(self):
        super().__init__()

        # Stage 1
        self.conv_in = nn.Conv2d(3, 128, kernel_size=3, padding=1)                # (B, 3, H, W) -> (B, 128, H, W)
        self.res1_1 = ResidualBlock(128, 128)                                     # (B, 128, H, W) -> (B, 128, H, W)
        self.res1_2 = ResidualBlock(128, 128)                                     # (B, 128, H, W) -> (B, 128, H, W)
        self.down1  = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0)     # (B, 128, H, W) -> (B, 128, H/2, W/2)

        # Stage 2
        self.res2_1 = ResidualBlock(128, 256)                                     # (B, 128, H/2, W/2) -> (B, 256, H/2, W/2)
        self.res2_2 = ResidualBlock(256, 256)                                     # (B, 256, H/2, W/2) -> (B, 256, H/2, W/2)
        self.down2  = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0)     # (B, 256, H/2, W/2) -> (B, 256, H/4, W/4)

        # Stage 3
        self.res3_1 = ResidualBlock(256, 512)                                     # (B, 256, H/4, W/4) -> (B, 512, H/4, W/4)
        self.res3_2 = ResidualBlock(512, 512)                                     # (B, 512, H/4, W/4) -> (B, 512, H/4, W/4)
        self.down3  = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0)     # (B, 512, H/4, W/4) -> (B, 512, H/8, W/8)

        # Bottleneck blocks (H/8, W/8)
        self.res4_1 = ResidualBlock(512, 512)
        self.res4_2 = ResidualBlock(512, 512)
        self.res4_3 = ResidualBlock(512, 512)
        self.attn    = AttentionBlock(512)
        self.res4_4 = ResidualBlock(512, 512)
        self.norm    = nn.GroupNorm(32, 512)
        self.act     = nn.SiLU()

        # Projection to params (kept identical to your original)
        # 3x3 with padding=1 keeps spatial size; then 1x1 to 8 channels; split -> mean/logvar (4 + 4)
        self.to_param_pre = nn.Conv2d(512, 8, kernel_size=3, padding=1)           # (B, 512, H/8, W/8) -> (B, 8, H/8, W/8)
        self.to_param     = nn.Conv2d(8, 8, kernel_size=1, padding=0)             # (B, 8, H/8, W/8) -> (B, 8, H/8, W/8)

    def _maybe_downsample_pad(self, x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        # Asymmetric padding before any stride-2 conv (same as your original forward)
        if getattr(layer, 'stride', None) == (2, 2):
            # Pad: (left, right, top, bottom) -> zeros on right & bottom
            x = F.pad(x, (0, 1, 0, 1))
        return x

    def forward(self, x: torch.Tensor, noise: torch.Tensor, return_features: bool = False):
        """
        x:     (B, 3,  H,  W)
        noise: (B, 4,  H/8, W/8)  ~ N(0, 1) or zeros if deterministic
        return:
          if return_features == False:
            (z, mean, log_variance)
          else:
            (z, mean, log_variance, feats)
            where feats is (B, 512, H/8, W/8), right before projection to mean/logvar
        """

        # Stage 1
        x = self.conv_in(x)
        x = self.res1_1(x)
        x = self.res1_2(x)
        x = self._maybe_downsample_pad(x, self.down1)
        x = self.down1(x)

        # Stage 2
        x = self.res2_1(x)
        x = self.res2_2(x)
        x = self._maybe_downsample_pad(x, self.down2)
        x = self.down2(x)

        # Stage 3
        x = self.res3_1(x)
        x = self.res3_2(x)
        x = self._maybe_downsample_pad(x, self.down3)
        x = self.down3(x)

        # Bottleneck
        x = self.res4_1(x)
        x = self.res4_2(x)
        x = self.res4_3(x)
        x = self.attn(x)
        x = self.res4_4(x)
        x = self.norm(x)
        x = self.act(x)

        # Expose this as the feature map for latent template matching
        feats = x  # (B, 512, H/8, W/8)

        # Project to params
        x = self.to_param_pre(x)   # (B, 8, H/8, W/8)
        x = self.to_param(x)       # (B, 8, H/8, W/8)

        # Split into mean/log-variance
        mean, log_variance = torch.chunk(x, 2, dim=1)  # (B, 4, H/8, W/8) each

        # Clamp log var for numeric stability (same bounds you used)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()

        # Reparameterize with provided noise
        z = mean + stdev * noise

        # Stable Diffusion scaling constant (as in your original)
        z = z * 0.18215

        if return_features:
            latent_map = self.project_latents_to_2d(mean, log_variance, reduce="l2")
            return z, mean, log_variance, feats, latent_map  
        return z, mean, log_variance

    # --- ADD THIS NEW METHOD ---
    @torch.no_grad()
    def project_latents_to_2d(
        self,
        mean: torch.Tensor,          # (B, 4, H/8, W/8)
        log_variance: torch.Tensor,  # (B, 4, H/8, W/8)
        reduce: Literal["l2", "l1", "mean", "kl"] = "l2",
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Returns a single-channel latent activation map per item (B, 1, H/8, W/8).

        reduce:
          - "l2":  sqrt(sum_c mean^2)            (default; robust & visually clean)
          - "l1":  sum_c |mean|
          - "mean": average over channels
          - "kl":  per-location KL(mean, logvar) to N(0, I), then sum over channels
        """
        if reduce == "kl":
            # KL per channel: 0.5 * (mu^2 + sigma^2 - logvar - 1)
            var = torch.exp(log_variance)
            kl = 0.5 * (mean.pow(2) + var - log_variance - 1.0)
            m = kl.sum(dim=1, keepdim=True)  # (B,1,H/8,W/8)
        elif reduce == "l1":
            m = mean.abs().sum(dim=1, keepdim=True)
        elif reduce == "mean":
            m = mean.mean(dim=1, keepdim=True)
        else:  # "l2"
            m = torch.sqrt(torch.clamp(mean.pow(2).sum(dim=1, keepdim=True), min=eps))
        return m

    def _maybe_downsample_pad(self, x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        # [KEEP as-is]
        if getattr(layer, 'stride', None) == (2, 2):
            x = F.pad(x, (0, 1, 0, 1))
        return x
    

class VQVAE_Encoder(nn.Sequential):

    def __init__(self, latent_dim=128):
        super().__init__(
            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            ResidualBlock(128, 128),
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            ResidualBlock(128, 128),
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height / 2, Width / 2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            # (Batch_Size, 128, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            ResidualBlock(128, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            ResidualBlock(256, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 4, Width / 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), 
            
            # (Batch_Size, 256, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            ResidualBlock(256, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            AttentionBlock(512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.GroupNorm(32, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.SiLU(), 

            # Because the padding=1, it means the width and height will increase by 2
            # Out_Height = In_Height + Padding_Top + Padding_Bottom
            # Out_Width = In_Width + Padding_Left + Padding_Right
            # Since padding = 1 means Padding_Top = Padding_Bottom = Padding_Left = Padding_Right = 1,
            # Since the Out_Width = In_Width + 2 (same for Out_Height), it will compensate for the Kernel size of 3
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8). 
            nn.Conv2d(512, 8, kernel_size=3, padding=1), 

            # (Batch_Size, 8, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8)
            nn.Conv2d(8, latent_dim, kernel_size=1, padding=0), 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channel, Height, Width)
        # noise: (Batch_Size, Out_Channels, Height / 8, Width / 8)
        # noise: (Batch_Size, 4, Height / 8, Width / 8)

        for module in self:

            if getattr(module, 'stride', None) == (2, 2):  # Padding at downsampling should be asymmetric (see #8)
                # Pad: (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom).
                # Pad with zeros on the right and bottom.
                # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Channel, Height + Padding_Top + Padding_Bottom, Width + Padding_Left + Padding_Right) = (Batch_Size, Channel, Height + 1, Width + 1)
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x)
        # (Batch_Size, 8, Height / 8, Width / 8) 
        return x