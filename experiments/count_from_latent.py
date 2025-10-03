# experiments/count_from_latent.py
import os
import json
import argparse
import glob
from typing import Tuple, Dict, List, Literal, Optional

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import csv

from models.vae import VAE
from tools.utils import select_device, set_seed, create_directory

# ----------------------------
# IO helpers
# ----------------------------
def load_image_rgb(path: str, resize: int) -> np.ndarray:
    """
    Load an image from 'path', convert BGR->RGB, and resize to (resize, resize).
    Returns an RGB uint8 array with shape (H, W, 3).
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_AREA)
    return img

def to_tensor_chw01(img_rgb_uint8: np.ndarray) -> torch.Tensor:
    """
    Convert an RGB uint8 image (H, W, 3) in [0,255] to a float32 tensor (C, H, W) in [0,1].
    """
    x = img_rgb_uint8.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # CHW
    return torch.from_numpy(x)

def standardize_channelwise_featmap(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Standardize a feature map per-channel over spatial dims: (z - mean)/std.
    z: (B, C, H, W)
    Returns a tensor with the same shape.
    """
    mean = z.mean(dim=(2,3), keepdim=True)
    std = z.std(dim=(2,3), keepdim=True)
    return (z - mean) / (std + eps)

def overlay_heatmap(image_rgb: np.ndarray, heat_upsampled01: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Colorize a heatmap (float [0,1]) and blend it over the original image.
    image_rgb: uint8 (H, W, 3)
    heat: float (H, W) in [0,1]
    alpha: blending factor for the heatmap.
    """
    heat_uint8 = (np.clip(heat_upsampled01, 0, 1) * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, heat_rgb, alpha, 0)
    return overlay

def draw_peaks_on_overlay(overlay: np.ndarray, peaks_hw: np.ndarray, scale_y: float, scale_x: float) -> np.ndarray:
    """
    Draw green circles on the overlay at predicted peak locations.
    peaks_hw: array of (h, w) indices in the downsampled response map.
    down_factor: scale factor to map peak indices to image pixels (e.g., 256/32 = 8).
    """
    out = overlay.copy()
    for (ph, pw) in peaks_hw:
        cy = int((ph + 0.5) * scale_y)
        cx = int((pw + 0.5) * scale_x)
        cv2.circle(out, (cx, cy), radius=15, color=(0,255,0), thickness=2)  # predicted
    return out

def draw_gt_boxes(overlay: np.ndarray,
                  boxes: List[Tuple[float,float,float,float]],
                  proto_box: Tuple[float,float,float,float] = None) -> np.ndarray:
    """
    Draw GT bounding boxes.
    - Red = all ground truth boxes
    - Blue = the GT box used for latent prototype (if provided)
    The proto_box comparison uses a small absolute tolerance to match floats.
    """
    out = overlay.copy()
    for (x1, y1, x2, y2) in boxes:
        color = (255, 0, 0)  # red
        if proto_box is not None:
            # compare with small tolerance
            if all(abs(a - b) < 1e-2 for a, b in zip([x1,y1,x2,y2], proto_box)):
                color = (0, 0, 255)  # blue
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return out

# ----------------------------
# CSV helpers
# ----------------------------
def read_yolo_csv(csv_path: str) -> Dict[str, List[Tuple[float,float,float,float]]]:
    """
    Read a YOLO-style CSV with columns:
      filename, class, cx, cy, w, h
    Returns a dict: {stem: [(cx,cy,w,h), ...], ...}
    where 'stem' is the filename without extension (matching image stems).
    """
    mapping: Dict[str, List[Tuple[float,float,float,float]]] = {}
    with open(csv_path, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            stem = os.path.splitext(row["filename"])[0]  # strip .txt
            cx, cy, w, h = map(float, (row["cx"], row["cy"], row["w"], row["h"]))
            mapping.setdefault(stem, []).append((cx, cy, w, h))
    return mapping

def yolo_to_xyxy_pixels(cx, cy, w, h, W, H):
    """
    Convert normalized YOLO (cx, cy, w, h) in [0,1] to pixel-space (x1,y1,x2,y2).
    W, H are the image width and height in pixels.
    """
    x1 = (cx - w/2.0) * W
    y1 = (cy - h/2.0) * H
    x2 = (cx + w/2.0) * W
    y2 = (cy + h/2.0) * H
    return (x1, y1, x2, y2)

# ----------------------------
# Feature selection
# ----------------------------
def _get_feature_map(
    model: VAE,
    x: torch.Tensor,
    feature_source: Literal["feats", "z", "mu"],
    noise: torch.Tensor
) -> torch.Tensor:
    """
    Returns the spatial feature map to correlate with the prototype.
    - "feats": last spatial map before μ/σ projection, shape (B, 512, H/8, W/8)
    - "z":     reparameterized latent (scaled by 0.18215), shape (B, 4, H/8, W/8)
    - "mu":    mean (μ) of the bottleneck distribution, shape (B, 4, H/8, W/8)
    The encoder is called with return_features accordingly.
    """
    model.eval()
    with torch.no_grad():
        if feature_source == "feats":
            # encoder requires noise; pass zeros to be safe/deterministic
            z, mu, logvar, feats = model.encoder(x, noise, return_features=True)
            return feats
        else:
            z, mu, logvar = model.encoder(x, noise, return_features=False)
            if feature_source == "z":
                return z
            elif feature_source == "mu":
                return mu
            else:
                raise ValueError(f"Unknown feature_source={feature_source}")

# ----------------------------
# Correlation + peak picking
def compute_response_map(z: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
    """
    Compute a normalized cross-correlation response map using conv2d.
    Steps:
      1) Standardize the feature map per-channel over spatial dims.
      2) Standardize the prototype per-channel over its (k,k) spatial dims.
      3) Convolve z_norm with proto as a single filter (1 output channel).
      4) Min-max normalize the response to [0,1] per-image.
    Args:
      z:     (1, C, h, w) feature map from the image
      proto: (C, k, k) prototype kernel
    Returns:
      resp01: (1,1,h,w) normalized response map in [0,1]
    """
    device = z.device
    B, C, h, w = z.shape           # B=1, C=channels, h/w=spatial size of fmap (e.g., 32x32 for 256/8)
    Ck, k, k2 = proto.shape        # proto has shape (C, k, k); Ck must equal C; k2==k
    assert C == Ck, f"Channel mismatch: z has {C}, prototype has {Ck}"

    # Standardize the input features channel-wise over spatial dims.
    # z_norm: (1, C, h, w)
    # mean/std: (1, C, 1, 1) broadcast over (h, w)
    z_norm = standardize_channelwise_featmap(z)

    # Standardize prototype per-channel across its spatial support.
    # proto before norm: (C, k, k)
    # proto.mean/std:    (C, 1, 1) broadcast over (k, k)
    # proto after norm:  (C, k, k)
    proto = (proto - proto.mean(dim=(1,2), keepdim=True)) / (proto.std(dim=(1,2), keepdim=True) + 1e-6)

    # Prepare weight as a conv filter: (out_channels=1, in_channels=C, k, k)
    # weight: (1, C, k, k)
    weight = proto.unsqueeze(0).to(device)  # (1, C, k, k)

    # Same padding to keep spatial size: pad = k//2 (assumes odd k)
    pad = k // 2

    # Convolution:
    # input:  z_norm  (1, C, h, w)
    # weight: weight  (1, C, k, k)
    # stride=1, padding=pad  -> output: (1, 1, h, w)
    resp = torch.nn.functional.conv2d(z_norm, weight=weight, stride=1, padding=pad)  # (1, 1, h, w)

    # Per-image (per-sample) min-max normalization to [0,1]
    # resp_min/max: (1, 1, 1, 1) reduced over (h, w)
    # resp01:       (1, 1, h, w)
    resp_min = resp.amin(dim=(2,3), keepdim=True)
    resp_max = resp.amax(dim=(2,3), keepdim=True)
    resp01 = (resp - resp_min) / (resp_max - resp_min + 1e-8)
    return resp01


def peak_pick(resp01: torch.Tensor, thresh: float = 0.6, pool_ks: int = 3):
    """
    Detect local maxima in the response map using max-pooling non-maximum suppression.
    Args:
      resp01: tensor (1,1,h,w) in [0,1]   -> normalized response map
      thresh: keep peaks with response >= thresh
      pool_ks: window size for local max pooling (e.g., 3 or 5)
    Returns:
      peaks_hw: ndarray of shape (N, 2) with (h, w) indices (integers)
      vals:     ndarray of shape (N,) with corresponding response values
    """
    # Apply max-pooling to find the local maximum value in each neighborhood.
    # resp01: (1, 1, h, w)
    # pooled: (1, 1, h, w) -> each pixel contains the local max in its pool_ks×pool_ks neighborhood
    pooled = torch.nn.functional.max_pool2d(resp01, kernel_size=pool_ks, stride=1, padding=pool_ks//2)

    # Boolean mask of peak candidates:
    # - resp01 >= pooled  -> only true where the pixel is equal to the local max
    # - resp01 >= thresh  -> filter by minimum response threshold
    # is_peak: (1, 1, h, w), dtype=bool
    is_peak = (resp01 >= pooled) & (resp01 >= thresh)

    # Get coordinates of all peaks
    # peaks: (N, 4) tensor, each row = [batch_idx, channel_idx, h_idx, w_idx]
    peaks = is_peak.nonzero(as_tuple=False)

    if peaks.numel() == 0:
        # If no peaks found, return empty arrays with correct shapes
        return np.zeros((0,2), dtype=np.int32), np.zeros((0,), dtype=np.float32)

    # Extract row/col indices of peaks
    # peaks[:, 2] -> h indices, shape (N,)
    # peaks[:, 3] -> w indices, shape (N,)
    h_idx = peaks[:, 2].cpu().numpy()   # (N,)
    w_idx = peaks[:, 3].cpu().numpy()   # (N,)

    # Gather corresponding response values at those positions
    # resp01[0,0,h_idx,w_idx] -> tensor (N,)
    # vals: numpy array (N,)
    vals = resp01[0,0,h_idx, w_idx].cpu().numpy()

    # Stack (h_idx, w_idx) into array of shape (N,2)
    peaks_hw = np.stack([h_idx, w_idx], axis=1)

    return peaks_hw, vals


# ----------------------------
# Main
# ----------------------------
def main():
    """
    Entry point for counting objects by correlating a latent prototype with feature maps.
    Produces visual overlays and (optionally) MAE/RMSE metrics when GT CSV is provided.
    """
    ap = argparse.ArgumentParser(description="Count pineapples by matching latent template.")
    ap.add_argument("--images_dir", required=True, help="Folder with .png images (also used to resolve --image_stem)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--prototype_pt", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--csv", default=None, help="Path to FULL_UNIFIED_labels.csv for ground truth (optional)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resize_img", type=int, default=256)
    ap.add_argument("--thresh", type=float, default=0.6)
    ap.add_argument("--pool_ks", type=int, default=3)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--only_annotated", action="store_true", help="Evaluate only images present in CSV")
    # single-image options
    ap.add_argument("--use_prototype_image", action="store_true",
                    help="Use the same image recorded in latent_prototype_meta.json")
    ap.add_argument("--image_path", type=str, default=None, help="Run on a single explicit image path (.png)")
    ap.add_argument("--image_stem", type=str, default=None, help="Run on a single stem within images_dir")
    # NEW: feature source for inference
    ap.add_argument("--feature_source", type=str, default="auto",
                    choices=["auto", "feats", "z", "mu"],
                    help="Which map to correlate at inference. 'auto' uses the prototype's metadata if available.")

    args = ap.parse_args()

    # Set device, seeds, and make sure output directory exists.
    device = select_device(args.device)
    set_seed(args.seed, deterministic=True, cudnn_benchmark=False)
    create_directory(args.output_dir)

    # Load VAE model and checkpoint weights (strict=False to be tolerant to missing keys).
    model = VAE().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load prototype tensor and (optionally) its metadata JSON (same folder).
    proto = torch.load(args.prototype_pt, map_location="cpu").float()
    proto_dir = os.path.dirname(args.prototype_pt)
    meta_json = os.path.join(proto_dir, "latent_prototype_meta.json")
    proto_meta = None
    if os.path.isfile(meta_json):
        with open(meta_json, "r") as f:
            proto_meta = json.load(f)
        print(f"Loaded prototype meta: {meta_json}")
    else:
        print("WARNING: Prototype metadata not found; running without meta checks.")

    # Decide which feature source to use for correlation.
    # - "auto": respect the feature_source stored in the prototype meta if present, else default to "z".
    # - explicit choice: use the CLI value.
    if args.feature_source == "auto":
        if proto_meta and "feature_source" in proto_meta:
            feature_source = str(proto_meta["feature_source"])
            print(f"[auto] Using feature_source from meta: {feature_source}")
        else:
            feature_source = "z"  # fallback to legacy behavior
            print(f"[auto] No meta feature_source; falling back to: {feature_source}")
    else:
        feature_source = args.feature_source
        print(f"Using feature_source from CLI: {feature_source}")

    # Build list of candidate images to process and read the prototype's source box if available.
    img_paths: List[str] = []
    proto_box = None

    # 1) exact image via prototype meta (use the exact same image used to extract the prototype)
    if args.use_prototype_image:
        if not proto_meta:
            raise FileNotFoundError("--use_prototype_image is set but no latent_prototype_meta.json was found.")
        meta_img_path = proto_meta.get("image_path")
        if meta_img_path is None or not os.path.isfile(meta_img_path):
            # If meta stores only a stem, try resolving it inside images_dir.
            stem = os.path.splitext(os.path.basename(proto_meta.get("image_path","")))[0]
            candidate = os.path.join(args.images_dir, stem + ".png")
            if not os.path.isfile(candidate):
                raise FileNotFoundError("Could not resolve prototype image from metadata.")
            meta_img_path = candidate
        if "box_xyxy" in proto_meta:
            proto_box = tuple(proto_meta["box_xyxy"])
            print(f"Prototype was extracted from box: {proto_box}")
        img_paths = [meta_img_path]

    # 2) explicit single image path
    elif args.image_path is not None:
        if not os.path.isfile(args.image_path):
            raise FileNotFoundError(f"--image_path not found: {args.image_path}")
        img_paths = [args.image_path]

    # 3) specific stem within images_dir (resolve to images_dir/<stem>.png)
    elif args.image_stem is not None:
        candidate = os.path.join(args.images_dir, args.image_stem + ".png")
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"--image_stem '{args.image_stem}' not found in {args.images_dir}")
        img_paths = [candidate]

    # 4) fallback: batch process .png files in images_dir (optionally limited by --limit)
    else:
        img_paths = sorted(glob.glob(os.path.join(args.images_dir, "*.png")))
        if args.limit > 0:
            img_paths = img_paths[:args.limit]

    # Optionally load GT annotations from CSV and filter images if --only_annotated is set.
    gt_ann = {}
    if args.csv:
        gt_ann = read_yolo_csv(args.csv)
        if args.only_annotated:
            stems_with_ann = set(gt_ann.keys())
            img_paths = [p for p in img_paths if os.path.splitext(os.path.basename(p))[0] in stems_with_ann]

    # Convenience: cache prototype channel count to check compatibility later.
    proto_C = int(proto.shape[0])

    # Process each image: forward pass -> correlation -> peak picking -> visualization -> metrics
    down_factor = args.resize_img // 8  # used to map latent grid coords to image pixels
    preds, gts = [], []
    for path in img_paths:
        # Load and tensorize image
        img = load_image_rgb(path, resize=args.resize_img)
        x = to_tensor_chw01(img).unsqueeze(0).to(device)

        _, _, H, W = x.shape
        zH, zW = H // 8, W // 8
        # Deterministic noise for feats/mu; random noise for z if desired (to reflect sampling).
        if feature_source in ("feats", "mu"):
            noise = torch.zeros((1, 4, zH, zW), device=device)
        else:
            noise = torch.randn((1, 4, zH, zW), device=device)

        # Extract the chosen feature map: feats / z / mu
        with torch.no_grad():
            fmap = _get_feature_map(model, x, feature_source=feature_source, noise=noise)  # (1,C,h,w)

        # Early, explicit check that prototype and fmap have the same channel count.
        if fmap.shape[1] != proto_C:
            raise RuntimeError(
                f"Prototype channels ({proto_C}) != feature map channels ({fmap.shape[1]}). "
                f"Ensure extract and count use the SAME --feature_source. "
                f"(Prototype meta feature_source={proto_meta.get('feature_source') if proto_meta else 'unknown'})"
            )

        # Cross-correlate feature map with prototype and detect peaks.
        resp01 = compute_response_map(fmap, proto.to(device))
        peaks_hw, _ = peak_pick(resp01, thresh=args.thresh, pool_ks=args.pool_ks)
        count_pred = int(peaks_hw.shape[0])

        # If GT annotations are available for this image, convert them to pixel-space.
        stem = os.path.splitext(os.path.basename(path))[0]
        gt_boxes = []
        if args.csv and stem in gt_ann:
            for (cx, cy, w, h) in gt_ann[stem]:
                gt_boxes.append(yolo_to_xyxy_pixels(cx, cy, w, h, args.resize_img, args.resize_img))
        count_gt = len(gt_boxes) if args.csv else -1  # -1 means unknown / not computing metrics

        # Accumulate metrics only when GT exists.
        if count_gt >= 0:
            preds.append(count_pred)
            gts.append(count_gt)

        # Visualization: overlay heatmap + peaks (+ optional GT boxes).
        resp_np = resp01[0,0].cpu().numpy()
        #overlay = overlay_heatmap(img, cv2.resize(resp_np, (args.resize_img, args.resize_img)))
        #overlay = draw_peaks_on_overlay(overlay, peaks_hw, down_factor)

        # 1) Upsample heatmap to image size using NEAREST (no interpolation drift)
        heat_up = cv2.resize(
            resp_np,
            (args.resize_img, args.resize_img),
            interpolation=cv2.INTER_LINEAR#INTER_NEAREST
        )

        # 2) Map peak grid indices to image pixels using per-axis scale from actual fmap size
        h_grid, w_grid = resp_np.shape[:2]
        scale_y = args.resize_img / float(h_grid)
        scale_x = args.resize_img / float(w_grid)

        overlay = overlay_heatmap(img, heat_up)
        overlay = draw_peaks_on_overlay(overlay, peaks_hw, scale_y=scale_y, scale_x=scale_x)

        if gt_boxes:
            overlay = draw_gt_boxes(overlay, gt_boxes, proto_box=proto_box)

        # Save overlay image for inspection.
        out_path = os.path.join(args.output_dir, f"{stem}_overlay.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if count_gt >= 0:
            print(f"{stem}: pred={count_pred}, gt={count_gt}, saved={out_path}")
        else:
            print(f"{stem}: pred={count_pred}, saved={out_path}")

    # Metrics (only if we had GTs for >0 images and we processed more than 1 image)
    if len(gts) > 0 and len(img_paths) > 1:
        preds = np.array(preds)
        gts = np.array(gts)
        mae = np.mean(np.abs(preds - gts))
        rmse = np.sqrt(np.mean((preds - gts) ** 2))
        print(f"\n[Metrics] MAE={mae:.2f}  RMSE={rmse:.2f}")
        with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
            f.write(f"MAE={mae:.2f}\nRMSE={rmse:.2f}\n")

if __name__ == "__main__":
    main()
