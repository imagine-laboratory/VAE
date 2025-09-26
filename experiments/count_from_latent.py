# experiments/count_from_latent.py
import os
import json
import argparse
import glob
from typing import Tuple, Dict, List

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

def standardize_channelwise_featmap(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = z.mean(dim=(2,3), keepdim=True)
    std = z.std(dim=(2,3), keepdim=True)
    return (z - mean) / (std + eps)

def overlay_heatmap(image_rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    heat_uint8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, heat_rgb, alpha, 0)
    return overlay

def draw_peaks_on_overlay(overlay: np.ndarray, peaks_hw: np.ndarray, down_factor: int) -> np.ndarray:
    out = overlay.copy()
    for (ph, pw) in peaks_hw:
        cy = int((ph + 0.5) * down_factor)
        cx = int((pw + 0.5) * down_factor)
        cv2.circle(out, (cx, cy), radius=6, color=(0,255,0), thickness=2)  # predicted
    return out

def draw_gt_boxes(overlay: np.ndarray,
                  boxes: List[Tuple[float,float,float,float]],
                  proto_box: Tuple[float,float,float,float] = None) -> np.ndarray:
    """
    Draw GT bounding boxes.
    - Red = all ground truth boxes
    - Blue = the GT box used for latent prototype (if provided)
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
    mapping: Dict[str, List[Tuple[float,float,float,float]]] = {}
    with open(csv_path, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            stem = os.path.splitext(row["filename"])[0]  # strip .txt
            cx, cy, w, h = map(float, (row["cx"], row["cy"], row["w"], row["h"]))
            mapping.setdefault(stem, []).append((cx, cy, w, h))
    return mapping

def yolo_to_xyxy_pixels(cx, cy, w, h, W, H):
    x1 = (cx - w/2.0) * W
    y1 = (cy - h/2.0) * H
    x2 = (cx + w/2.0) * W
    y2 = (cy + h/2.0) * H
    return (x1, y1, x2, y2)

# ----------------------------
# Core
# ----------------------------
def compute_response_map(z: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
    device = z.device
    B, C, h, w = z.shape
    Ck, k, k2 = proto.shape
    assert C == Ck, f"Channel mismatch: z has {C}, prototype has {Ck}"

    z_norm = standardize_channelwise_featmap(z)
    proto = (proto - proto.mean(dim=(1,2), keepdim=True)) / (proto.std(dim=(1,2), keepdim=True) + 1e-6)

    weight = proto.unsqueeze(0).to(device)  # (1,C,k,k)
    pad = k // 2
    resp = torch.nn.functional.conv2d(z_norm, weight=weight, stride=1, padding=pad)  # (1,1,h,w)

    resp_min = resp.amin(dim=(2,3), keepdim=True)
    resp_max = resp.amax(dim=(2,3), keepdim=True)
    resp01 = (resp - resp_min) / (resp_max - resp_min + 1e-8)
    return resp01

def peak_pick(resp01: torch.Tensor, thresh: float = 0.6, pool_ks: int = 3):
    pooled = torch.nn.functional.max_pool2d(resp01, kernel_size=pool_ks, stride=1, padding=pool_ks//2)
    is_peak = (resp01 >= pooled) & (resp01 >= thresh)

    peaks = is_peak.nonzero(as_tuple=False)
    if peaks.numel() == 0:
        return np.zeros((0,2), dtype=np.int32), np.zeros((0,), dtype=np.float32)

    h_idx = peaks[:, 2].cpu().numpy()
    w_idx = peaks[:, 3].cpu().numpy()
    vals = resp01[0,0,h_idx, w_idx].cpu().numpy()
    peaks_hw = np.stack([h_idx, w_idx], axis=1)
    return peaks_hw, vals

# ----------------------------
# Main
# ----------------------------
def main():
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
    args = ap.parse_args()

    device = select_device(args.device)
    set_seed(args.seed, deterministic=True, cudnn_benchmark=False)
    create_directory(args.output_dir)

    # Load model
    model = VAE().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load prototype
    proto = torch.load(args.prototype_pt, map_location="cpu").float()
    print(f"Loaded prototype: {args.prototype_pt} shape={tuple(proto.shape)}")

    # Candidate image list
    img_paths: List[str] = []
    proto_box = None

    # 1) exact image via prototype meta
    if args.use_prototype_image:
        meta_json = os.path.join(os.path.dirname(args.prototype_pt), "latent_prototype_meta.json")
        if not os.path.isfile(meta_json):
            raise FileNotFoundError(f"--use_prototype_image is set but metadata not found at {meta_json}")
        with open(meta_json, "r") as f:
            meta = json.load(f)
        meta_img_path = meta.get("image_path")
        if meta_img_path is None or not os.path.isfile(meta_img_path):
            # Try resolving against images_dir using the stem
            stem = os.path.splitext(os.path.basename(meta.get("image_path","")))[0]
            candidate = os.path.join(args.images_dir, stem + ".png")
            if not os.path.isfile(candidate):
                raise FileNotFoundError("Could not resolve prototype image from metadata.")
            meta_img_path = candidate
        if "box_xyxy" in meta:
            proto_box = tuple(meta["box_xyxy"])
            print(f"Prototype was extracted from box: {proto_box}")
        img_paths = [meta_img_path]

    # 2) explicit image path
    elif args.image_path is not None:
        if not os.path.isfile(args.image_path):
            raise FileNotFoundError(f"--image_path not found: {args.image_path}")
        img_paths = [args.image_path]

    # 3) stem within images_dir
    elif args.image_stem is not None:
        candidate = os.path.join(args.images_dir, args.image_stem + ".png")
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"--image_stem '{args.image_stem}' not found in {args.images_dir}")
        img_paths = [candidate]

    # 4) fallback: batch over directory
    else:
        img_paths = sorted(glob.glob(os.path.join(args.images_dir, "*.png")))
        if args.limit > 0:
            img_paths = img_paths[:args.limit]

    # Load annotations if provided
    gt_ann = {}
    if args.csv:
        gt_ann = read_yolo_csv(args.csv)
        if args.only_annotated:
            stems_with_ann = set(gt_ann.keys())
            img_paths = [p for p in img_paths if os.path.splitext(os.path.basename(p))[0] in stems_with_ann]

    # Process images
    down_factor = args.resize_img // 8
    preds, gts = [], []
    for path in img_paths:
        img = load_image_rgb(path, resize=args.resize_img)
        x = to_tensor_chw01(img).unsqueeze(0).to(device)

        _, _, H, W = x.shape
        zH, zW = H // 8, W // 8
        noise = torch.randn((1, 4, zH, zW), device=device)
        with torch.no_grad():
            z, mu, logvar = model.encoder(x, noise)

        resp01 = compute_response_map(z, proto.to(device))
        peaks_hw, _ = peak_pick(resp01, thresh=args.thresh, pool_ks=args.pool_ks)
        count_pred = int(peaks_hw.shape[0])

        stem = os.path.splitext(os.path.basename(path))[0]
        gt_boxes = []
        if args.csv and stem in gt_ann:
            for (cx, cy, w, h) in gt_ann[stem]:
                gt_boxes.append(yolo_to_xyxy_pixels(cx, cy, w, h, args.resize_img, args.resize_img))
        count_gt = len(gt_boxes) if args.csv else -1  # -1 means unknown

        if count_gt >= 0:
            preds.append(count_pred)
            gts.append(count_gt)

        # Visualization
        resp_np = resp01[0,0].cpu().numpy()
        overlay = overlay_heatmap(img, cv2.resize(resp_np, (args.resize_img, args.resize_img)))
        overlay = draw_peaks_on_overlay(overlay, peaks_hw, down_factor)
        if gt_boxes:
            overlay = draw_gt_boxes(overlay, gt_boxes, proto_box=proto_box)

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
