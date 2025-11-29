"""Dataset class for BCS classification."""
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class BcsDataset(Dataset):
    """Dataset for BCS classification with ROI cropping and optional padding."""
    def __init__(self, csv_path, img_size=320, train=True, do_aug=False, crop_padding=None):
        self.df = pd.read_csv(csv_path)
        self.img_size = img_size
        self.train = train
        self.do_aug = do_aug
        self.crop_padding = float(crop_padding) if crop_padding is not None else 0.0

        tf = [A.Resize(img_size, img_size)]
        if train and do_aug:
            tf += [A.HorizontalFlip(p=0.5), A.Rotate(limit=10, p=0.5)]
        tf += [A.Normalize(), ToTensorV2()]
        self.tf = A.Compose(tf)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(r.image_path)[:, :, ::-1]
        
        if pd.notna(r.get("xmin", np.nan)):
            try:
                h, w = img.shape[:2]
                xmin, ymin, xmax, ymax = map(int, [r.xmin, r.ymin, r.xmax, r.ymax])
                
                if self.crop_padding > 0:
                    bbox_w = xmax - xmin
                    bbox_h = ymax - ymin
                    pad_w = int(bbox_w * self.crop_padding)
                    pad_h = int(bbox_h * self.crop_padding)
                    
                    xmin = max(0, xmin - pad_w)
                    xmax = min(w, xmax + pad_w)
                    ymin = max(0, ymin - pad_h)
                    ymax = min(h, ymax + pad_h)
                
                xmin = max(0, min(xmin, w - 1))
                xmax = max(1, min(xmax, w))
                ymin = max(0, min(ymin, h - 1))
                ymax = max(1, min(ymax, h))
                
                if xmax > xmin and ymax > ymin:
                    img = img[ymin:ymax, xmin:xmax]
            except Exception:
                pass

        x = self.tf(image=img)["image"]
        y = int(r.bcs_5class)
        return x, y
