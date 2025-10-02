# add the root directory to the Python path
import sys
import os
# check if the grand parent directory is already in sys.path to avoid duplicates
if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.vae import decoder,encoder
import model_converter
import torch
from data.datasets import PineappleDataset
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm   # <-- progress bar
# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)
ckpt_path = 'weights/v1-5-pruned-emaonly.ckpt'
state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

encoder_model = encoder.VAE_Encoder().to(device)
encoder_model.load_state_dict(state_dict["encoder"], strict=True)

decoder_model = decoder.VAE_Decoder().to(device)
decoder_model.load_state_dict(state_dict["decoder"], strict=True)

root_folder = 'data/folds' # here is were the subfoledrs with the images are
target_folder = 'data/vae_reconstructions' # here is where the reconstructions will be saved in subfolders as well
os.makedirs(target_folder, exist_ok=True)
subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]

encoder_model.eval()
decoder_model.eval()

batch_size = 8  # adjust to your GPU memory
num_workers = 0  # adjust for your machine

for subfolder in subfolders:
    # Create an output subfolder with the same name
    out_sub = os.path.join(target_folder, os.path.basename(subfolder.rstrip(os.sep)))
    os.makedirs(out_sub, exist_ok=True)

    # Load ALL images from this subfolder (disable test split; use the training slice at 100%)
    dataset = PineappleDataset(
        test_txt=None,
        path=subfolder,
        train=True,          # use the train split
        val=False,
        train_ratio=1.0,     # 100% of images go to "train"
        val_ratio=0.0,
        resize_img=256,
        augment=False        # no augmentations for recon
    )
    if len(dataset) == 0:
        print(f"[WARN] No images found in: {subfolder}")
        continue

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    with torch.no_grad():
        # wrap loader in tqdm for progress bar
        for batch in tqdm(loader, total=len(loader), desc=f"Processing {os.path.basename(subfolder)}"):
            # (B, 3, H, W) in [0,1]
            x = torch.from_numpy(np.stack(batch['image'])).to(device, dtype=torch.float32)

            # If your VAE expects inputs in [-1,1] (typical for SD VAEs), do this:
            x_scaled = x * 2.0 - 1.0

            # Prepare noise for encoder (B, 4, H/8, W/8)
            B, _, H, W = x_scaled.shape
            noise = torch.randn((B, 4, H // 8, W // 8), device=device, dtype=torch.float32)

            # Forward pass
            latent, mean, logvar = encoder_model(x_scaled, noise)
            reconstruction = decoder_model(latent)  # expected shape (B, 3, H, W), typically in [-1,1]

            # Back to [0,1]
            recon_01 = (reconstruction.clamp(-1, 1) + 1.0) / 2.0
            recon_01 = recon_01.clamp(0, 1).cpu().numpy()  # (B, 3, H, W)

            # Save each image in the batch with original base names
            for i, idx in enumerate(batch['idx']):
                src_path = dataset.images[int(idx)]
                base = os.path.splitext(os.path.basename(src_path))[0]
                out_path = os.path.join(out_sub, f"{base}.png")

                # Convert CHW [0,1] -> HWC uint8
                img = (np.transpose(recon_01[i], (1, 2, 0)) * 255.0).round().astype(np.uint8)
                Image.fromarray(img).save(out_path, format="PNG")

    print(f"[OK] Saved reconstructions for {subfolder} -> {out_sub}")
# forward pass for the VAE example
# use the PineappleDataset to load images from a folder
'''batch_size, _, height, width = x.shape
# The encoder expects noise with shape (Batch_Size, 4, Height/8, Width/8).
noise = torch.randn((batch_size, 4, height // 8, width // 8), device=x.device)
latent, mean, logvar = encoder_model(x, noise)
reconstruction = decoder_model(latent)'''
#return reconstruction, mean, logvar