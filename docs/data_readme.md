### Splits
- Created `data/train.csv` and `data/val.csv` via `scripts/split_manifest.py`
- Strategy: stratified by `bcs_5class`, test_size=0.2, random_state=42
- Rationale: fixed split for comparable experiments
