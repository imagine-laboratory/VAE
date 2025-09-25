#!/usr/bin/env python3
"""
visualize_latents.py

Run the encoder on template images and save a per-image 2D latent visualization (.jpg).

Usage:
  python visualize_latents.py \
    --templates_dir /data/ffallas/generative/VAE/template_crops_dir \
    --weights /data/ffallas/generative/VAE/output/checkpoints/betaKL@0.001/weights_ck_397.pt \
    --out_dir /data/ffallas/generative/VAE/latent_maps \
    --resize 256 \
    --batch_size 8 \
    --reduce l2 \
    --glob "*.png"

Notes:
- 'reduce' can be one of: l2, l1, mean, kl (see encoder.project_latents_to_2d).
- Images are assumed to be RGB and will be resized to --resize x --resize, normalized to [0,1].
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# --- Try reasonable class names from models/vae.py without forcing the user to edit this file
def load_vae_class():
    from importlib import import_module
    m = import_module("models.vae")
    # Try common class names
    for name in ("VAE", "SimpleVAE", "PineappleVAE"):
        if hasattr(m, name):
            return getattr(m, name)
    # Fallback: pick the first nn.Module subclass exported (last resort)
    import torch.nn as nn
    for attr in dir(m):
        obj = getattr(m, attr)
        try:
            if isinstance(obj, type) and issubclass(obj, nn.Module):
                return obj
        except Exception:
            pass
    raise RuntimeError("Could not find a VAE class in models/vae.py. Please export your model as class VAE.")

def list_images(root: str, pattern: str) -> List[str]:
    exts = [pattern]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, ext)))
    files = sorted(files)
    return files

def read_rgb_resize(path: str, size: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    return img

def save_gray_jpg(array_2d: np.ndarray, out_path: str) -> None:
    """
    array_2d: H x W, values will be min-max normalized to [0,255] (uint8).
    """
    a = array_2d.astype(np.float32)
    mn, mx = a.min(), a.max()
    if mx > mn:
        a = (a - mn) / (mx - mn)
    else:
        a = np.zeros_like(a)
    a = (a * 255.0).clip(0, 255).astype(np.uint8)
    # Save as grayscale
    cv2.imwrite(out_path, a)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates_dir", type=str, default="/data/ffallas/generative/VAE/template_crops_dir",
                        help="Directory with template crops (e.g., /data/.../template_crops_dir).")
    parser.add_argument("--weights", type=str, required=True,
                        help="Checkpoint path to load (e.g., .../weights_ck_397.pt).")
    parser.add_argument("--out_dir", type=str, default="/data/ffallas/generative/VAE/latent_maps",
                        help="Directory to save latent visualizations (.jpg).")
    parser.add_argument("--resize", type=int, default=256,
                        help="Input size used during training/inference (square).")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--glob", type=str, default="*.png",
                        help='Glob for templates (e.g., "*.png" or "*.png").')
    parser.add_argument("--reduce", type=str, default="l2", choices=["l2", "l1", "mean", "kl"],
                        help="Reduction used to project latent parameters to a 2D map.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Build model
    VAEClass = load_vae_class()
    model = VAEClass()  # Adjust if your constructor needs args
    model.eval()

    # --- Load weights
    # Accept both full checkpoints and pure state_dicts
    ckpt = torch.load(args.weights, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] Missing keys: {missing}", file=sys.stderr)
    if unexpected:
        print(f"[warn] Unexpected keys: {unexpected}", file=sys.stderr)

    model.to(args.device)

    # --- Collect images
    paths = list_images(args.templates_dir, args.glob)
    if not paths:
        print(f"No images found in {args.templates_dir} with pattern {args.glob}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(paths)} images.")

    # --- Inference loop (batched)
    B = args.batch_size
    H = W = args.resize
    h8, w8 = H // 8, W // 8

    with torch.no_grad():
        for i in range(0, len(paths), B):
            batch_paths = paths[i:i+B]

            imgs = [read_rgb_resize(p, args.resize) for p in batch_paths]
            x = torch.from_numpy(np.stack(imgs, axis=0))  # (B, 3, H, W)
            x = x.to(args.device)

            # Deterministic encoding: noise zeros with correct spatial size
            noise = torch.zeros((x.size(0), 4, h8, w8), device=args.device, dtype=x.dtype)

            # We call the encoder directly to get the latent map
            # Expect the model to expose .encoder with the modified forward
            assert hasattr(model, "encoder"), "Model must expose .encoder"
            out = model.encoder(x, noise, return_features=True)

            # Backward-compatible tuple sizes: (z, mean, logvar, feats, latent_map)
            if len(out) == 5:
                z, mean, logvar, feats, latent_map = out
            else:
                # If your local encoder hasn't been updated yet, emulate latent_map here:
                z, mean, logvar, feats = out
                latent_map = model.encoder.project_latents_to_2d(mean, logvar, reduce=args.reduce)

            # Upsample latent map to input size
            m = F.interpolate(latent_map, size=(H, W), mode="bilinear", align_corners=False)  # (B,1,H,W)
            m = m.squeeze(1).detach().cpu().numpy()  # (B,H,W)

            # Save one .jpg per image
            for p, m2d in zip(batch_paths, m):
                base = Path(p).stem
                out_path = os.path.join(args.out_dir, f"{base}_latent.jpg")
                save_gray_jpg(m2d, out_path)

            print(f"[{i+len(batch_paths)}/{len(paths)}] saved latent maps.")

    print("Done.")

if __name__ == "__main__":
    main()


# From repo root
#python visualize_latents.py --weights /data/ffallas/generative/VAE/output/checkpoints/betaKL@0.001/weights_ck_397.pt
