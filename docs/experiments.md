## 2025-01-XX — Dataset split v2 (Train/Val/Test)
- From: data/manifest.csv (53,566 rows)
- Split: 70/15/15 stratified by `bcs_5class`, seed=42
- Files: 
  - data/train.csv (37,496 samples, 70%)
  - data/val.csv (8,035 samples, 15%) - used for model selection during training
  - data/test.csv (8,035 samples, 15%) - held out for final evaluation only
- Script: `scripts/split_manifest_3way.py`
- Note: Test set should only be evaluated on final chosen model(s) using `scripts/evaluate_test.py`

## 2025-10-06 — Dataset split v1 (deprecated)
- Split: 80/20 (train/val only)
- Files: data/train.csv (42,852), data/val.csv (10,714)
- Note: Replaced by v2 with separate test set.

## 2025-10-06 — E1 Baseline (smoke)
Data: train.csv / val.csv (v1), img=320
Model: EfficientNet-B0 (pretrained), CE+LS(0.05), bs=32, epochs=2, lr=3e-4, seed=42
Val: Acc = <printout>
Artifacts: outputs/exp_E1_2025-10-06/
Notes: pipeline OK; proceed to full baseline.

## 2025-10-06 — Speed tuning
Config: mps, img=224, bs=32, workers=2, cv2 loader.
Observation: step frequency improved; next—cache 224px ROI crops, then try bs=48.

## 2025-10-08 — E1 Baseline (smoke, full train once)
**Commit:** 0c02f3c
**Config:** `configs/default.yaml`  
- Device: mps
- Data: `data/train.csv` / `data/val.csv`, img=224
- Model: EfficientNet-B0 (pretrained=False), num_classes=5
- Train: bs=32, epochs=1, lr=3e-4, seed=42, workers=4, prefetch=2, persistent_workers=true

**Results (Val):**  
- Accuracy: `0.3219`  
- Notes: runtime ~ `15mins`; Baseline accuracy is above random (0.20 for 5 classes) but low; next we’ll enable pretrained weights and longer training. Also plan to add macro-F1 & per-class recall.

**Artifacts:** `outputs/exp_E1_2025-10-08_17-45-54` (contains `best_acc.pt`, `config.yaml`)

## 2025-10-08 — E1b Pretrained baseline (6 epochs)
**Commit:** 0c02f3c
**Config:** img=224, bs=32, mps, workers=2, pretrained=True
**Results (Val):** Acc=0.8799; (macro-F1, per-class recall pending)
**Runtime:** ~1h30
**Notes:** Strong baseline; next add macro-F1/recall, confusion matrix, and early stopping to save time.
**Artifacts:** outputs/exp_E1_2025-10-08_18-16-34/
