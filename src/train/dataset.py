"""Dataset class for BCS classification."""
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class BcsDataset(Dataset):
    """
    Loads images/labels from a CSV with columns:
      image_path, bcs_5class (0..4), xmin, ymin, xmax, ymax
    - Crops to ROI if bbox present, else uses full image.
    - Resizes to img_size, normalizes, returns tensor.
    """
    def __init__(self, csv_path, img_size=320, train=True, do_aug=False, skip_resize=False):
        """
        Args:
            csv_path: Path to CSV with image paths and labels
            img_size: Target image size (default 320, but 224 is standard)
            train: Whether this is training set
            do_aug: Whether to apply augmentation
            skip_resize: If True, skip resizing (assumes images are already correct size)
        """
        self.df = pd.read_csv(csv_path)
        self.img_size = img_size
        self.train = train
        self.do_aug = do_aug
        self.skip_resize = skip_resize

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
                    img = img[ymin:ymax, xmin:xmax]
            except Exception:
                pass  # fall back to full image if anything goes wrong

        x = self.tf(image=img)["image"]
        y = int(r.bcs_5class)  # 0..4
        return x, y
