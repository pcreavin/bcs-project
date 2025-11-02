### Splits
- Created `data/train.csv`, `data/val.csv`, and `data/test.csv` via `scripts/split_manifest_3way.py`
- Strategy: 70% train / 15% validation / 15% test, stratified by `bcs_5class`, random_state=42
- Rationale: fixed split for comparable experiments, held-out test set for final evaluation

**Split sizes:**
- Train: 37,496 samples (70.0%)
- Validation: 8,035 samples (15.0%) - used for model selection, early stopping
- Test: 8,035 samples (15.0%) - held out for final evaluation only

**Note:** Validation set is used during training for hyperparameter tuning and model selection. Test set should only be evaluated on final chosen model(s).
