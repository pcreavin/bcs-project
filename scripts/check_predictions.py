#!/usr/bin/env python3
"""Check predictions for specific samples."""
import torch
import yaml
from src.train.dataset import BcsDataset
from src.models import create_model

# Load model
with open("outputs/roi_robustness_padding_05/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

model_cfg = cfg.get("model", {})
data_cfg = cfg.get("data", {})

backbone = model_cfg.get("backbone", "efficientnet_b0")
num_classes = int(model_cfg.get("num_classes", 5))
pretrained = bool(model_cfg.get("pretrained", True))
finetune_mode = model_cfg.get("finetune_mode", "full")
img_size = int(data_cfg.get("img_size", 224))

model = create_model(backbone=backbone, num_classes=num_classes, pretrained=pretrained, finetune_mode=finetune_mode)
model.load_state_dict(torch.load("outputs/roi_robustness_padding_05/best_model.pt", map_location="cpu"))
model.eval()

test_ds = BcsDataset("data/test.csv", img_size=img_size, train=False, do_aug=False)
class_names = ["3.25", "3.5", "3.75", "4.0", "4.25"]

for sample_num in [21, 16, 23, 24]:
    with torch.no_grad():
        image_tensor, true_label = test_ds[sample_num]
        image_tensor = image_tensor.unsqueeze(0)
        output = model(image_tensor)
        pred_class = output.argmax(dim=1).item()
    
    true_bcs = class_names[true_label]
    pred_bcs = class_names[pred_class]
    print(f"Sample {sample_num}: True={true_bcs}, Pred={pred_bcs}")

