"""Dataset class for BCS classification."""
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import random

class BcsDataset(Dataset):
    """
    Loads images/labels from a CSV with columns:
      image_path, bcs_5class (0..4), xmin, ymin, xmax, ymax
    - Crops to ROI if bbox present, else uses full image.
    - Supports crop jitter: random padding around ROI bbox for robustness.
    - Resizes to img_size, normalizes, returns tensor.
    """
    def __init__(self, csv_path, img_size=320, train=True, do_aug=False, skip_resize=False, crop_jitter=0.0):
        """
        Args:
            csv_path: Path to CSV with image paths and labels
            img_size: Target image size (default 320, but 224 is standard)
            train: Whether this is training set
            do_aug: Whether to apply augmentation
            skip_resize: If True, skip resizing (assumes images are already correct size)
            crop_jitter: If > 0, add random padding around ROI bbox (as fraction of bbox size)
                        Only applies during training when bbox is present
        """
        self.df = pd.read_csv(csv_path)
        self.img_size = img_size
        self.train = train
        self.do_aug = do_aug
        self.skip_resize = skip_resize
        self.crop_jitter = crop_jitter if train else 0.0  # Only use jitter during training

        tf = []
        if not skip_resize:
            tf.append(A.Resize(img_size, img_size))
        if train and do_aug:
            tf += [A.HorizontalFlip(p=0.5), A.Rotate(limit=10, p=0.5)]
        tf += [A.Normalize(), ToTensorV2()]
        self.tf = A.Compose(tf) if tf else A.Compose([A.Normalize(), ToTensorV2()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(r.image_path)[:, :, ::-1]  # BGR->RGB
        
        # ROI crop (if bbox present) - skip if using preprocessed images
        if not self.skip_resize and pd.notna(r.get("xmin", np.nan)):
            try:
                h, w = img.shape[:2]
                xmin, ymin, xmax, ymax = map(int, [r.xmin, r.ymin, r.xmax, r.ymax])
                xmin = max(0, min(xmin, w - 1))
                xmax = max(1, min(xmax, w))
                ymin = max(0, min(ymin, h - 1))
                ymax = max(1, min(ymax, h))
                
                if xmax > xmin and ymax > ymin:
                    # Apply crop jitter if enabled during training
                    if self.crop_jitter > 0:
                        # Calculate bbox dimensions
                        bbox_w = xmax - xmin
                        bbox_h = ymax - ymin
                        
                        # Random jitter amount (0 to crop_jitter fraction)
                        jitter_w = random.uniform(0, self.crop_jitter) * bbox_w
                        jitter_h = random.uniform(0, self.crop_jitter) * bbox_h
                        
                        # Expand bbox with random padding (but stay within image bounds)
                        xmin = max(0, int(xmin - jitter_w / 2))
                        ymin = max(0, int(ymin - jitter_h / 2))
                        xmax = min(w, int(xmax + jitter_w / 2))
                        ymax = min(h, int(ymax + jitter_h / 2))
                    
                    img = img[ymin:ymax, xmin:xmax]
            except Exception:
                pass  # fall back to full image if anything goes wrong

        x = self.tf(image=img)["image"]
        y = int(r.bcs_5class)  # 0..4
        return x, y
