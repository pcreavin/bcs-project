#!/usr/bin/env python3
"""Preprocess images: crop ROI and resize to 224x224 for faster training.

This script:
1. Loads images from train.csv, val.csv, test.csv
2. Crops ROI if bbox exists
3. Resizes to 224x224
4. Saves preprocessed images
5. Creates new CSV files pointing to preprocessed images

Usage:
    python scripts/preprocess_images.py
"""
import pandas as pd
import cv2
import pathlib
from tqdm import tqdm
import argparse


def preprocess_split(csv_path: str, output_dir: pathlib.Path, split_name: str, img_size: int):
    """Preprocess images for one split (train/val/test)."""
    print(f"\nProcessing {split_name} split...")
    df = pd.read_csv(csv_path)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failed = 0
    
    for idx, r in tqdm(df.iterrows(), total=len(df), desc=f"{split_name}"):
        # Load original image
        img = cv2.imread(r.image_path)
        if img is None:
            print(f"Warning: Could not load {r.image_path}")
            failed += 1
            continue
        
        # Crop ROI if bbox exists
        if pd.notna(r.get("xmin", None)):
            try:
                h, w = img.shape[:2]
                x1, y1, x2, y2 = map(int, [r.xmin, r.ymin, r.xmax, r.ymax])
                x1 = max(0, min(x1, w - 1))
                x2 = max(1, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(1, min(y2, h))
                if x2 > x1 and y2 > y1:
                    img = img[y1:y2, x1:x2]
            except Exception as e:
                # Fall back to full image
                pass
        
        # Resize to img_size x img_size
        img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        
        # Save preprocessed image
        # Use original filename to preserve uniqueness
        original_name = pathlib.Path(r.image_path).name
        out_path = output_dir / original_name
        cv2.imwrite(str(out_path), img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        # Create new row with preprocessed image path
        new_row = {
            "image_path": str(out_path),
            "bcs_float": r.bcs_float,
            "bcs_5class": int(r.bcs_5class),
            # No bbox needed for preprocessed images
            "xmin": "",
            "ymin": "",
            "xmax": "",
            "ymax": "",
        }
        rows.append(new_row)
    
    # Save new CSV
    new_csv = pd.DataFrame(rows)
    output_csv = output_dir.parent / f"{split_name}_processed.csv"
    new_csv.to_csv(output_csv, index=False)
    
    print(f"  ✓ Processed {len(rows)} images")
    print(f"  ✓ Saved to {output_dir}")
    print(f"  ✓ CSV saved to {output_csv}")
    if failed > 0:
        print(f"  ⚠ Failed to process {failed} images")
    
    return output_csv


def main():
    parser = argparse.ArgumentParser(description="Preprocess images for faster training")
    parser.add_argument("--train-csv", type=str, default="data/train.csv")
    parser.add_argument("--val-csv", type=str, default="data/val.csv")
    parser.add_argument("--test-csv", type=str, default="data/test.csv")
    parser.add_argument("--output-dir", type=str, default="data/processed_224")
    parser.add_argument("--img-size", type=int, default=224, help="Target square size for resized images")
    args = parser.parse_args()
    
    output_base = pathlib.Path(args.output_dir)
    
    print("=" * 60)
    print("IMAGE PREPROCESSING")
    print("=" * 60)
    print(f"Output directory: {output_base}")
    print("This will:")
    print("  1. Crop ROI from images (if bbox exists)")
    print(f"  2. Resize to {args.img_size}x{args.img_size}")
    print("  3. Save preprocessed images")
    print("  4. Create new CSV files for training")
    print("=" * 60)
    
    # Process each split
    train_csv = preprocess_split(args.train_csv, output_base / "train", "train", args.img_size)
    val_csv = preprocess_split(args.val_csv, output_base / "val", "val", args.img_size)
    test_csv = preprocess_split(args.test_csv, output_base / "test", "test", args.img_size)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nPreprocessed CSVs:")
    print(f"  Train: {train_csv}")
    print(f"  Val:   {val_csv}")
    print(f"  Test:  {test_csv}")
    print("\nTo use preprocessed images:")
    print("  Update configs to point to:")
    print(f"    train: {train_csv}")
    print(f"    val:   {val_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()







