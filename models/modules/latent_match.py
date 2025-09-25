import math
from typing import Dict, Literal, Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image, ImageDraw
import numpy as np


# ---------------------------
# 1) Feature extraction from your encoder
# ---------------------------
@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    img: torch.Tensor,                 # (1, 3, H, W) in [0..1]
    use_pre_z: bool = True,            # True: use pre-z features (B, 512, H/8, W/8); False: use mean (B, 4, H/8, W/8)
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Pass an image through your encoder and return:
      - feats  (512 channels) if use_pre_z=True (recommended for discrimination)
      - mean   (4 channels)   if use_pre_z=False (latent "mu")
    Assumes your encoder forward(..., return_features=True) returns (z, mean, logvar, feats, latent_map).
    """
    if device is None:
        device = img.device
    encoder.eval()

    B, C, H, W = img.shape
    # Deterministic pass: zeros noise. Your encoder handles asymmetric padding internally.
    noise = torch.zeros((B, 4, math.ceil(H / 8), math.ceil(W / 8)), device=device)
    z, mean, logvar, feats, _latent2d = encoder(img.to(device), noise, return_features=True)

    return feats if use_pre_z else mean


# ---------------------------
# 2) Template summarization (vector prototype OR small 2D kernel)
# ---------------------------
@torch.no_grad()
def make_template_descriptor(
    template_feats: torch.Tensor,           # (1, C, ht, wt)
    mode: Literal["vector", "kernel"] = "vector",
    center_frac: float = 0.6,               # for "vector": average over central window (fraction of side length)
    kernel_size: int = 9,                   # for "kernel": central crop size in feature space
    l2_normalize: bool = True
) -> torch.Tensor:
    """
    Returns:
      - mode="vector" -> (C,) prototype vector (L2-normalized if l2_normalize)
      - mode="kernel" -> (C, k, k) 2D kernel (per-channel L2-normalized if l2_normalize)
    """
    assert template_feats.dim() == 4 and template_feats.size(0) == 1
    _, C, Hf, Wf = template_feats.shape

    if mode == "vector":
        # Average pooling on the central window
        ch = max(1, int(round(Hf * center_frac)))
        cw = max(1, int(round(Wf * center_frac)))
        top = max(0, (Hf - ch) // 2)
        left = max(0, (Wf - cw) // 2)
        window = template_feats[:, :, top:top + ch, left:left + cw]  # (1, C, ch, cw)
        proto = window.mean(dim=(0, 2, 3))                           # (C,)
        if l2_normalize:
            proto = F.normalize(proto, dim=0, eps=1e-8)
        return proto  # (C,)

    # mode == "kernel": take a central k×k crop
    k = min(kernel_size, Hf, Wf)
    top = (Hf - k) // 2
    left = (Wf - k) // 2
    kernel = template_feats[:, :, top:top + k, left:left + k].squeeze(0).contiguous()  # (C, k, k)
    if l2_normalize:
        # Normalize per channel to stabilize correlation
        kernel = kernel.view(C, -1)
        kernel = F.normalize(kernel, dim=1, eps=1e-8)
        kernel = kernel.view(C, k, k)
    return kernel  # (C, k, k)


# ---------------------------
# 3) Apply descriptor on the image feature map (similarity/correlation)
# ---------------------------
@torch.no_grad()
def similarity_map_cosine_vector(
    feats: torch.Tensor,        # (1, C, Hf, Wf)
    proto: torch.Tensor         # (C,)
) -> torch.Tensor:
    """
    Cosine similarity per spatial position (1x1) across channels.
    Recommended when the template is summarized into a single vector (channel pattern).
    Returns a heatmap in [0..1], shape (1, 1, Hf, Wf).
    """
    assert feats.dim() == 4 and feats.size(0) == 1
    C = feats.size(1)
    assert proto.dim() == 1 and proto.numel() == C

    # L2-normalize features across channels and the prototype
    nf = F.normalize(feats, dim=1, eps=1e-8)                 # (1, C, Hf, Wf)
    p  = F.normalize(proto, dim=0, eps=1e-8).view(1, C, 1, 1)  # (1, C, 1, 1)
    sim = (nf * p).sum(dim=1, keepdim=True)                  # (1, 1, Hf, Wf)
    return (sim + 1.0) * 0.5  # map [-1, 1] to [0, 1]


@torch.no_grad()
def correlation_map_ncc_kernel(
    feats: torch.Tensor,            # (1, C, Hf, Wf)
    kernel: torch.Tensor            # (C, k, k)
) -> torch.Tensor:
    """
    Cosine similarity via conv2d numerator and per-patch L2 norm denominator.
    Returns (1,1,Hout,Wout) in [0,1].
    """
    assert feats.dim() == 4 and feats.size(0) == 1
    B, C, Hf, Wf = feats.shape
    Ck, k, k2 = kernel.shape
    assert Ck == C and k == k2

    # Ensure kernel is L2-normalized per-channel (flattened)
    ker = kernel.view(C, -1)
    ker = torch.nn.functional.normalize(ker, dim=1, eps=1e-8).view(C, k, k)

    # Numerator: dot(kernel, patch) with conv2d
    # weight shape for conv2d: (out_channels=1, in_channels=C, k, k)
    weight = ker.unsqueeze(0).contiguous()           # (1, C, k, k)
    num = torch.nn.functional.conv2d(feats, weight)  # (1, 1, Hout, Wout)

    # Denominator: L2 norm of each image patch (C*k*k)
    patches = torch.nn.functional.unfold(feats, kernel_size=k, stride=1, padding=0)  # (1, C*k*k, L)
    denom = patches.norm(p=2, dim=1, keepdim=True) + 1e-8                             # (1,1,L)
    Hout = Hf - k + 1
    Wout = Wf - k + 1
    denom = denom.view(1, 1, Hout, Wout)                                              # (1,1,Hout,Wout)

    cos = num / denom                      # cosine similarity in [-1, 1]
    return (cos + 1.0) * 0.5               # to [0, 1]



# ---------------------------
# 5) Post-processing: threshold, peak picking, and NMS
# ---------------------------
@torch.no_grad()
def _heatmap_topk(
    heatmap: torch.Tensor,  # (1,1,H,W) in [0,1]
    topk: int,
    peak_thresh: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast peak selection: take global top-K scores without scanning all >=thresh.
    Returns (idx_2d [N,2], scores [N]) with N<=topk and scores>=peak_thresh.
    Works on CPU to avoid long GPU sorts on huge maps.
    """
    hm = heatmap.squeeze(0).squeeze(0).contiguous()   # (H, W)
    H, W = hm.shape
    flat = hm.view(-1).cpu()                          # move to CPU for fast topk
    k = min(topk, flat.numel())
    vals, inds = torch.topk(flat, k, largest=True, sorted=True)  # (k,)
    # threshold
    keep = vals >= peak_thresh
    if keep.sum() == 0:
        return torch.empty(0, 2, dtype=torch.long), torch.empty(0)
    vals = vals[keep]
    inds = inds[keep]
    ys = (inds // W).to(torch.long)
    xs = (inds %  W).to(torch.long)
    idx_2d = torch.stack([ys, xs], dim=1)            # (N,2)
    return idx_2d, vals


def _iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    IoU between a (1, 4) and b (N, 4). Boxes are xyxy (float).
    """
    x1 = torch.max(a[:, 0], b[:, 0])
    y1 = torch.max(a[:, 1], b[:, 1])
    x2 = torch.min(a[:, 2], b[:, 2])
    y2 = torch.min(a[:, 3], b[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)

    union = area_a + area_b - inter + 1e-8
    return inter / union


def _nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vanilla NMS implemented in PyTorch (no torchvision dependency).
    Input:
      boxes  (N, 4) xyxy
      scores (N,)
    Output:
      boxes_kept, scores_kept (ordered by score descending)
    """
    if boxes.numel() == 0:
        return boxes, scores

    # Sort by score desc
    order = torch.argsort(scores, descending=True)
    boxes = boxes[order]
    scores = scores[order]

    keep_indices = []
    while boxes.size(0) > 0:
        keep_indices.append(0)
        if boxes.size(0) == 1:
            break
        ious = _iou_xyxy(boxes[0].unsqueeze(0), boxes[1:])
        keep_mask = ious <= iou_thresh
        # Keep the first (current max) and those with IoU below threshold
        boxes = torch.cat([boxes[0:1], boxes[1:][keep_mask]], dim=0)
        scores = torch.cat([scores[0:1], scores[1:][keep_mask]], dim=0)

    # Recompute final order by score (already kept in descending order)
    return boxes, scores


# --- FAST PEAK SELECTION (local maxima) + TOP-K ---

@torch.no_grad()
def _heatmap_localmax_topk(
    heatmap: torch.Tensor,   # (1,1,H,W) in [0,1]
    topk: int,
    peak_thresh: float,
    pool_kernel: int = 3,
    dilation: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Keep only local maxima via max-pooling, then take global top-K.
    Returns (idx_2d [N,2], scores [N]), N<=topk.
    """
    hm = heatmap.squeeze(0).squeeze(0)   # (H, W)
    H, W = hm.shape

    pad = dilation * (pool_kernel - 1) // 2
    pooled = torch.nn.functional.max_pool2d(
        hm.unsqueeze(0).unsqueeze(0),
        kernel_size=pool_kernel,
        stride=1,
        padding=pad,
        dilation=dilation
    ).squeeze()

    # local maxima mask
    localmax = (hm >= pooled) & (hm >= peak_thresh)
    if not localmax.any():
        return torch.empty(0, 2, dtype=torch.long), torch.empty(0)

    ys, xs = torch.nonzero(localmax, as_tuple=True)
    vals = hm[ys, xs]

    k = min(topk, vals.numel())
    vals, order = torch.topk(vals, k, largest=True, sorted=True)
    ys = ys[order]
    xs = xs[order]
    idx_2d = torch.stack([ys, xs], dim=1)
    return idx_2d, vals


# --- FAST NMS (uses torchvision if present) ---

def _nms_xyxy_fast(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prefer torchvision.ops.nms; fallback to a reasonably fast pure-PyTorch NMS.
    Inputs on CUDA will run NMS on CUDA if torchvision is available.
    """
    try:
        from torchvision.ops import nms as tv_nms  # type: ignore
        keep = tv_nms(boxes, scores, iou_thresh)
        return boxes[keep], scores[keep]
    except Exception:
        pass  # fallback below

    # Pure-PyTorch fallback (vectorized loop without tensor concatenations)
    if boxes.numel() == 0:
        return boxes, scores

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    order = torch.argsort(scores, descending=True)
    keep_idx = []
    while order.numel() > 0:
        i = order[0]
        keep_idx.append(i.item())
        if order.numel() == 1:
            break

        rest = order[1:]

        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])

        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-8)

        keep_mask = iou <= iou_thresh
        order = rest[keep_mask]

    keep = torch.tensor(keep_idx, device=boxes.device, dtype=torch.long)
    return boxes[keep], scores[keep]


# --- DOWNSAMPLE HEATMAP OPTION (optional speed knob) ---

@torch.no_grad()
def _maybe_downsample_heatmap(heatmap: torch.Tensor, stride: int) -> Tuple[torch.Tensor, float]:
    """
    Optionally downsample the heatmap before peak picking.
    Returns (heatmap_ds, scale) where scale maps coords back: orig_coord = (coord_ds + 0.5)*stride - 0.5
    """
    if stride <= 1:
        return heatmap, 1.0
    H, W = heatmap.shape[-2:]
    Hds = max(1, H // stride)
    Wds = max(1, W // stride)
    hm_ds = torch.nn.functional.interpolate(heatmap, size=(Hds, Wds), mode="bilinear", align_corners=False)
    return hm_ds, float(stride)


# --- REPLACEMENT: heatmap_to_boxes (fast) ---

@torch.no_grad()
def heatmap_to_boxes(
    heatmap: torch.Tensor,          # (1,1,Hm,Wm) in feature space
    downsample: int,                # 8 for your encoder
    kernel_size: Optional[int],     # if mode="kernel": k; else None
    topk: int = 100,
    peak_thresh: float = 0.6,
    nms_iou: float = 0.3,
    box_scale: float = 1.2,
    pool_kernel: int = 3,           # NEW: local-max window in heatmap space
    pool_dilation: int = 1,         # NEW: dilated local-max if needed
    heat_stride: int = 1            # NEW: downsample heatmap before peak picking (e.g., 2 or 4)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast path:
      1) (optional) downsample heatmap for speed
      2) local-max suppression via max-pooling
      3) global top-K
      4) vectorized NMS (torchvision if available)
    """
    device = heatmap.device

    # 1) Optional downsample of heatmap
    hm, stride = _maybe_downsample_heatmap(heatmap, heat_stride)  # (1,1,Hds,Wds), stride>=1

    # 2)+3) Local maxima + Top-K
    idx_2d_ds, scores = _heatmap_localmax_topk(
        hm, topk=topk, peak_thresh=peak_thresh,
        pool_kernel=pool_kernel, dilation=pool_dilation
    )
    if idx_2d_ds.numel() == 0:
        return torch.empty(0, 4, device=device), torch.empty(0, device=device)

    # Map DS heatmap coords back to original heatmap coords (float)
    ys_ds = idx_2d_ds[:, 0].to(device).float()
    xs_ds = idx_2d_ds[:, 1].to(device).float()
    # center-preserving mapping
    xs = (xs_ds + 0.5) * stride - 0.5
    ys = (ys_ds + 0.5) * stride - 0.5

    scores = scores.to(device)

    # 4) Build boxes in IMAGE space
    cx = (xs + 0.5) * downsample
    cy = (ys + 0.5) * downsample

    if kernel_size is None:
        bw = bh = float(downsample * box_scale)
    else:
        bw = bh = float(kernel_size * downsample * box_scale)

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    # Vectorized NMS
    boxes_nms, scores_nms = _nms_xyxy_fast(boxes, scores, iou_thresh=nms_iou)
    return boxes_nms, scores_nms



# ---------------------------
# I/O and visualization helpers
# ---------------------------
def load_image(path: str, size_divisible_by: int = 1) -> Tuple[Image.Image, torch.Tensor]:
    """
    Load an RGB image as PIL and as a float32 torch Tensor in [0..1], shape (1, 3, H, W).
    We do NOT enforce H, W to be multiples of 8; your encoder handles padding.
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return img, t


def draw_boxes_on_pil(img: Image.Image, boxes_xyxy: torch.Tensor, color=(255, 0, 0), width: int = 3) -> Image.Image:
    """
    Draw axis-aligned XYXY boxes on a PIL image and return a copy.
    """
    im = img.copy()
    draw = ImageDraw.Draw(im)
    for b in boxes_xyxy.cpu().tolist():
        x1, y1, x2, y2 = b
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return im


def save_heatmap_overlay(
    img: Image.Image,
    heatmap: torch.Tensor,              # (1, 1, Hm, Wm) in [0, 1]
    downsample: int,
    out_path: str
):
    """
    Resize the feature-space heatmap to the original image size and save a simple overlay (grayscale).
    """
    hm = heatmap.squeeze().cpu().numpy()
    Hm, Wm = hm.shape

    hm_img = Image.fromarray((hm * 255).astype(np.uint8), mode="L")
    hm_img = hm_img.resize(img.size, resample=Image.BILINEAR)

    # Gray overlay blend
    hm_rgb = Image.merge("RGB", (hm_img, hm_img, hm_img))
    blended = Image.blend(img.convert("RGB"), hm_rgb, alpha=0.5)
    blended.save(out_path)
