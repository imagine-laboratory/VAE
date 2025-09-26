# tools/latent_template_extract.py
import os
import json
import argparse
import csv
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

import numpy as np
import cv2

from models.vae import VAE
from tools.utils import select_device, set_seed, create_directory

# ----------------------------
# Helpers
# ----------------------------
def load_image_rgb(path: str, resize: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_AREA)
    return img

def to_tensor_chw01(img_rgb_uint8: np.ndarray) -> torch.Tensor:
    x = img_rgb_uint8.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # CHW
    return torch.from_numpy(x)

def read_yolo_csv(csv_path: str) -> Dict[str, List[Tuple[float,float,float,float]]]:
    """
    Returns dict: {basename (without .png): [(cx,cy,w,h), ...], ...}
    The CSV uses .txt filenames; the corresponding image is same stem with .png
    """
    mapping: Dict[str, List[Tuple[float,float,float,float]]] = {}
    with open(csv_path, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            txtname = row["filename"]  # e.g., 5b84....txt
            stem = os.path.splitext(txtname)[0]
            cx, cy, w, h = map(float, (row["cx"], row["cy"], row["w"], row["h"]))
            mapping.setdefault(stem, []).append((cx, cy, w, h))
    return mapping

def yolo_to_xyxy_pixels(cx, cy, w, h, W, H):
    x = (cx - w/2.0) * W
    y = (cy - h/2.0) * H
    ww = w * W
    hh = h * H
    return (x, y, x + ww, y + hh)

def standardize_channelwise(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    t: (C, k, k) or (1, C, k, k)
    """
    orig_shape = t.shape
    if t.dim() == 3:
        t = t.unsqueeze(0)  # (1,C,k,k)
    mean = t.mean(dim=(2,3), keepdim=True)
    std = t.std(dim=(2,3), keepdim=True)
    t = (t - mean) / (std + eps)
    return t.view(orig_shape)

# ----------------------------
# Core
# ----------------------------
def extract_latent_roi_template(
    model: VAE,
    image_path: str,
    box_xyxy_img: Tuple[float,float,float,float],
    resize_img: int = 256,
    roi_size: int = 9,
) -> Tuple[torch.Tensor, dict]:
    """
    Returns:
      proto: (C, roi_size, roi_size) float32 tensor
      meta: dict with info for reproducibility
    """
    model.eval()
    device = next(model.parameters()).device

    # Load & preprocess
    img = load_image_rgb(image_path, resize=resize_img)
    H = W = resize_img
    x = to_tensor_chw01(img).unsqueeze(0).to(device)  # (1,3,H,W)

    # Build noise to match your VAE.forward
    _, _, height, width = x.shape
    zH, zW = height // 8, width // 8  # per your VAE code
    noise = torch.randn((1, 4, zH, zW), device=device)

    # Call encoder directly to get latent z
    with torch.no_grad():
        z, mu, logvar = model.encoder(x, noise)  # z: (1,Cz,h,w)

    _, Cz, h, w = z.shape
    spatial_scale = h / float(H)  # Expect 1/8

    # ROIAlign expects coords in input-image pixels; we pass spatial_scale
    x1, y1, x2, y2 = box_xyxy_img
    rois = torch.tensor([[0, x1, y1, x2, y2]], dtype=torch.float32, device=device)  # (num_rois, 5)

    roi_feat = roi_align(
        input=z,           # (N,C,h,w)
        boxes=rois,        # (num_rois,5) in image coords
        output_size=(roi_size, roi_size),
        spatial_scale=spatial_scale,
        sampling_ratio=-1,
        aligned=True
    )  # (num_rois, Cz, roi_size, roi_size)

    proto = roi_feat[0].detach().cpu()  # (Cz,k,k)

    # Optional: simple channel-wise standardization
    proto = standardize_channelwise(proto)

    meta = dict(
        image_path=image_path,
        box_xyxy=[float(x1), float(y1), float(x2), float(y2)],
        resize_img=resize_img,
        roi_size=roi_size,
        latent_shape=[int(Cz), int(h), int(w)],
        spatial_scale=float(spatial_scale),
        note="Channel-wise standardized",
    )
    return proto, meta

def main():
    ap = argparse.ArgumentParser(description="Extract a latent (RoIAlign) template from annotated images.")
    ap.add_argument("--images_dir", required=True, help="Folder with .png images")
    ap.add_argument("--csv", required=True, help="YOLO CSV with normalized cx,cy,w,h")
    ap.add_argument("--checkpoint", required=False, default=None, help="Path to VAE checkpoint (best.pt). If omitted, uses random weights (not recommended).")
    ap.add_argument("--output_dir", required=True, help="Where to save prototype .pt + metadata.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--resize_img", type=int, default=256)
    ap.add_argument("--roi_size", type=int, default=9)
    ap.add_argument("--max_crops", type=int, default=1, help="How many crops to average into the prototype (1 = single crop).")
    ap.add_argument("--image_stem", type=str, default=None, help="Optional: force using a specific image stem (without extension).")
    args = ap.parse_args()

    device = select_device(args.device)
    set_seed(args.seed, deterministic=True, cudnn_benchmark=False)
    create_directory(args.output_dir)

    model = VAE().to(device)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("WARNING: No checkpoint provided. Proceeding with randomly initialized VAE (prototype quality will be poor).")

    ann = read_yolo_csv(args.csv)

    # Determine which annotated samples to use
    stems = [args.image_stem] if args.image_stem else list(ann.keys())
    used = 0
    protos = []
    proto_metas = []

    for stem in stems:
        img_path = os.path.join(args.images_dir, stem + ".png")
        if not os.path.isfile(img_path):
            continue
        for (cx,cy,w,h) in ann[stem]:
            x1,y1,x2,y2 = yolo_to_xyxy_pixels(cx, cy, w, h, args.resize_img, args.resize_img)
            proto, meta = extract_latent_roi_template(model, img_path, (x1,y1,x2,y2),
                                                      resize_img=args.resize_img, roi_size=args.roi_size)
            protos.append(proto)     # (C,k,k) CPU
            proto_metas.append(meta)
            used += 1
            print(f"Added crop from {stem}  box={x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}")
            if used >= args.max_crops:
                break
        if used >= args.max_crops:
            break

    if len(protos) == 0:
        raise RuntimeError("No prototypes extracted. Check CSV, paths, and stems.")

    # Build final prototype (mean if >1)
    with torch.no_grad():
        proto_stack = torch.stack(protos, dim=0)  # (N,C,k,k)
        proto_mean = proto_stack.mean(dim=0)      # (C,k,k)

    # Save
    proto_path = os.path.join(args.output_dir, "latent_prototype.pt")
    meta_path = os.path.join(args.output_dir, "latent_prototype_meta.json")
    torch.save(proto_mean, proto_path)
    with open(meta_path, "w") as f:
        json.dump(dict(
            **{"num_crops": len(protos)},
            **proto_metas[0], # base meta
        ), f, indent=2)

    print(f"Saved prototype tensor to: {proto_path}")
    print(f"Saved metadata to:       {meta_path}")

if __name__ == "__main__":
    main()
