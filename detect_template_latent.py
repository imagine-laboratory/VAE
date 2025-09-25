import argparse
import os
import torch
from torch import nn

from models.vae import VAE
from models.modules.latent_match import (
    extract_features,
    make_template_descriptor,
    similarity_map_cosine_vector,
    correlation_map_ncc_kernel,
    heatmap_to_boxes,
    load_image,
    draw_boxes_on_pil,
    save_heatmap_overlay,
)

# ---------------------------
# Flexible checkpoint loader
# ---------------------------

def load_weights_flex(model: nn.Module, ckpt_path: str):
    """
    Load model weights from common checkpoint formats:
      - raw state_dict
      - dicts with keys like: 'state_dict', 'model', 'model_state', 'model_state_dict'
    Uses strict=False to tolerate non-matching keys.
    """
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        for key in ["state_dict", "model", "model_state", "model_state_dict"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)} (showing up to 10) -> {missing[:10]}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)} (showing up to 10) -> {unexpected[:10]}")
    print("[OK] Weights loaded with strict=False.")


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Latent template detection (vector or kernel) — no multiscale yet.")
    ap.add_argument("--ckpt", default="output/checkpoints/betaKL@0.001/weights_ck_397.pt", help="Path to VAE checkpoint (state_dict or wrapped dict).")
    ap.add_argument("--image", default="template_crops_dir/0525dbe8a8034863b154c0f21dd58bdc_original.png", help="Target image path.")
    ap.add_argument("--template", default="template_crops_dir/0525dbe8a8034863b154c0f21dd58bdc__templ_01.png", help="Template image path.")

    # Representation
    ap.add_argument("--use_pre_z", action="store_true",
                    help="Use pre-z feats (512 channels) instead of mean(4). Recommended.")
    ap.add_argument("--mode", choices=["vector", "kernel"], default="vector", help="Template summarization mode.")
    ap.add_argument("--center_frac", type=float, default=0.6, help="(vector) central window fraction for pooling.")
    ap.add_argument("--kernel_size", type=int, default=9, help="(kernel) k for central crop in feature space.")

    # Post-processing
    ap.add_argument("--peak_thresh", type=float, default=0.6, help="Heatmap threshold [0..1].")
    ap.add_argument("--topk", type=int, default=200, help="Max number of peaks to consider.")
    ap.add_argument("--nms_iou", type=float, default=0.3, help="IoU for NMS.")
    ap.add_argument("--box_scale", type=float, default=1.2, help="Scale factor for box size.")

    # I/O
    ap.add_argument("--out_dir", default="latent_maps", help="Output directory for heatmap/overlays.")
    ap.add_argument("--prefix", default="detect", help="Filename prefix.")
    ap.add_argument("--device", default="cuda", help="'cuda' or 'cpu'.")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # 1) Model
    model = VAE().to(device)
    load_weights_flex(model, args.ckpt)
    encoder = model.encoder
    print("[OK] Model ready.")

    # 2) Load images (no need to force multiples of 8; your encoder pads internally)
    pil_img,  img_t = load_image(args.image)
    pil_tmp,  tmp_t = load_image(args.template)
    print(f"[OK] Images loaded: img {pil_img.size}, tmp {pil_tmp.size}")

    img_t = img_t.to(device)
    tmp_t = tmp_t.to(device)

    # 3) Extract features
    feats_img = extract_features(encoder, img_t, use_pre_z=args.use_pre_z, device=device)   # (1, C, Hf, Wf)
    feats_tmp = extract_features(encoder, tmp_t, use_pre_z=args.use_pre_z, device=device)   # (1, C, ht, wt)
    print(f"[OK] Features extracted: img {feats_img.shape}, tmp {feats_tmp.shape}")

    # 4) Summarize template and compute heatmap
    if args.mode == "vector":
        proto = make_template_descriptor(feats_tmp, mode="vector", center_frac=args.center_frac)  # (C,)
        heat  = similarity_map_cosine_vector(feats_img, proto)                                    # (1, 1, Hf, Wf)
        k_for_boxes = None
    else:
        kernel = make_template_descriptor(feats_tmp, mode="kernel", kernel_size=args.kernel_size) # (C, k, k)
        heat   = correlation_map_ncc_kernel(feats_img, kernel)                                    # (1, 1, Hout, Wout)
        k_for_boxes = kernel.shape[-1]
    print(f"[OK] Heatmap computed: {heat.shape}")

    # 5) Post-processing (heatmap -> boxes in image coords)
    boxes_xyxy, scores = heatmap_to_boxes(
        heatmap=heat,
        downsample=8,                      # your encoder outputs H/8, W/8
        kernel_size=k_for_boxes,
        topk=args.topk,
        peak_thresh=args.peak_thresh,
        nms_iou=args.nms_iou,
        box_scale=args.box_scale,
        heat_stride=4
    )
    print(f"[OK] {len(boxes_xyxy)} boxes extracted.")

    # 6) Save outputs
    base = f"{args.prefix}"
    print(f"[INFO] Saving outputs to {args.out_dir} with base '{base}'")

    overlay_path = os.path.join(args.out_dir, f"{base}_heat_overlay.jpg")
    save_heatmap_overlay(pil_img, heat, downsample=8, out_path=overlay_path)
    print("[OK] Heatmap overlay saved.")

    boxed = draw_boxes_on_pil(pil_img, boxes_xyxy, color=(255, 0, 0), width=3)
    boxes_path = os.path.join(args.out_dir, f"{base}_boxes.jpg")
    boxed.save(boxes_path)
    print("[OK] Boxes image saved.")

     # Save boxes to text file

    txt_path = os.path.join(args.out_dir, f"{base}_detections.txt")
    with open(txt_path, "w") as f:
        for (x1, y1, x2, y2), s in zip(boxes_xyxy.cpu().tolist(), scores.cpu().tolist()):
            f.write(f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},{s:.4f}\n")
    print("[OK] Detections saved.")
    print("\n\n\nSummary of outputs:")

    print(f"[OK] Heatmap overlay  -> {overlay_path}")
    print(f"[OK] Boxes image      -> {boxes_path}")
    print(f"[OK] Detections (txt) -> {txt_path}")


if __name__ == "__main__":
    main()


#python detect_template_latent.py --use_pre_z --mode vector --center_frac 0.9 --peak_thresh 0.9 --topk 80 --nms_iou 0.5 --box_scale 30.0 --out_dir latent_maps --prefix demo_vec
#python detect_template_latent.py --use_pre_z --mode kernel --kernel_size 9 --center_frac 0.6 --peak_thresh 0.6 --topk 200 --nms_iou 0.3 --box_scale 1.2 --out_dir latent_maps --prefix demo_veck9