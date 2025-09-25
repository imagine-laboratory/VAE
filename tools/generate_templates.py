# =============================
# File: tools/generate_templates.py
# =============================
from __future__ import annotations
import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm


SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class BBox:
    cx: float  # normalized [0,1]
    cy: float  # normalized [0,1]
    w: float   # normalized [0,1]
    h: float   # normalized [0,1]

    def to_xyxy(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert normalized YOLO (cx,cy,w,h) to absolute (x1,y1,x2,y2) in pixels, clipped to image bounds."""
        cx_px = self.cx * width
        cy_px = self.cy * height
        w_px = self.w * width
        h_px = self.h * height
        x1 = int(round(cx_px - w_px / 2))
        y1 = int(round(cy_px - h_px / 2))
        x2 = int(round(cx_px + w_px / 2))
        y2 = int(round(cy_px + h_px / 2))
        # clip
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        # ensure non-empty
        if x2 <= x1:
            x2 = min(width, x1 + 1)
        if y2 <= y1:
            y2 = min(height, y1 + 1)
        return x1, y1, x2, y2


def _load_annotations_csv(labels_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    # Normalize column names and types
    expected_cols = {"filename", "class", "cx", "cy", "w", "h"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    # Some CSVs might include stray whitespace in filenames
    df["filename"] = df["filename"].astype(str).str.strip()
    return df


def _index_ann_by_image(df: pd.DataFrame) -> Dict[str, List[BBox]]:
    grouped: Dict[str, List[BBox]] = {}
    for _, row in df.iterrows():
        # The CSV stores label filenames such as 'abc.txt'; image file will share base name
        label_name = Path(row["filename"]).name
        base = Path(label_name).stem  # without .txt
        bbox = BBox(cx=float(row["cx"]), cy=float(row["cy"]), w=float(row["w"]), h=float(row["h"]))
        grouped.setdefault(base, []).append(bbox)
    return grouped


def _find_image_for_base(dataset_dir: Path, base: str) -> Optional[Path]:
    # Look in dataset_dir (non-recursive) first, then recursive
    for p in [*dataset_dir.glob(base + ".*"), *dataset_dir.rglob(base + ".*")]:
        if p.suffix.lower() in SUPPORTED_EXTS and p.is_file():
            return p
    return None


def _paste_centered(canvas: Image.Image, crop: Image.Image) -> Image.Image:
    W, H = canvas.size
    w, h = crop.size
    x = (W - w) // 2
    y = (H - h) // 2
    canvas.paste(crop, (x, y))
    return canvas


def make_centered_template(original_img: Image.Image, bbox: BBox) -> Image.Image:
    """Create a black canvas the same size as original_img and paste the bbox crop centered."""
    W, H = original_img.size
    x1, y1, x2, y2 = bbox.to_xyxy(W, H)
    crop = original_img.crop((x1, y1, x2, y2))
    canvas = Image.new("RGB", (W, H), color=(0, 0, 0))
    return _paste_centered(canvas, crop)


def generate_templates(
    dataset_dir: Path,
    labels_csv: Path,
    output_dir: Path,
    num_images: int,
    max_subimages_per_image: int,
    shuffle: bool = True,
    seed: int = 42,
) -> int:
    """Generate centered pineapple templates.

    Returns the number of images written.
    """
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_annotations_csv(labels_csv)
    ann_by_image = _index_ann_by_image(df)
    bases = list(ann_by_image.keys())
    if shuffle:
        random.shuffle(bases)

    written = 0
    selected_bases = bases[:num_images] if num_images > 0 else bases

    for base in tqdm(selected_bases, desc="Images", unit="img"):
        img_path = _find_image_for_base(dataset_dir, base)
        if img_path is None:
            # Skip silently but inform via tqdm postfix
            tqdm.write(f"[WARN] No image found for base '{base}'")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            tqdm.write(f"[WARN] Failed to open {img_path}: {e}")
            continue

        bboxes = ann_by_image.get(base, [])
        if not bboxes:
            continue

        # Decide how many subimages to create for this image
        k = len(bboxes) if max_subimages_per_image <= 0 else min(max_subimages_per_image, len(bboxes))
        # Choose the first k (stable) — or sample randomly for diversity
        chosen = bboxes if max_subimages_per_image <= 0 else random.sample(bboxes, k)

        for idx, bbox in enumerate(chosen):
            templ = make_centered_template(img, bbox)
            out_name = f"{base}__templ_{idx:02d}.png"
            templ.save(output_dir / out_name)
            written += 1

    return written


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate template images by centering cropped pineapples on a black canvas of the original size."
        )
    )
    p.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("/data/ffallas/datasets/vae/FULL_UNIFIED"),
        help="Root directory where the source images live.",
    )
    p.add_argument(
        "--labels_csv",
        type=Path,
        default=Path("/data/ffallas/datasets/vae/FULL_UNIFIED_labels.csv"),
        help="CSV with columns: filename,class,cx,cy,w,h (YOLO normalized).",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/data/ffallas/generative/VAE/template_crops_dir"),
        help="Where to save generated template images.",
    )
    p.add_argument(
        "--num_images",
        type=int,
        default=100,
        help="How many distinct source images to process (<= 0 means all).",
    )
    p.add_argument(
        "--max_subimages_per_image",
        type=int,
        default=2,
        help="How many subimages (cropped pineapples) to create per image (<= 0 means all).",
    )
    p.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Do not shuffle image order; process deterministically.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (used when sampling subimages).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    written = generate_templates(
        dataset_dir=args.dataset_dir,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        num_images=args.num_images,
        max_subimages_per_image=args.max_subimages_per_image,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )
    print(f"Done. Wrote {written} template image(s) to {args.output_dir}")


if __name__ == "__main__":
    main()



