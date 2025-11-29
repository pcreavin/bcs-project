# Ensemble Experiments Status

## ✅ COMPLETED

### 1. Classification + Ordinal (2 models, Rule-based)
- **Script**: `evaluate_ensemble.py`
- **Models**: 
  - Baseline: `ablation_full_enhanced_b0_224_jitter`
  - Ordinal: `ordinal_b0_224_jitter_weighted`
- **Method**: Rule-based (same → use it, diff by 1 → classification, diff >1 → ordinal)
- **Status**: ✅ Complete (val + test sets)
- **Results**: 
  - Val: 96.09% acc, 96.22% macro-F1
  - Test: 96.25% acc, 96.41% macro-F1

---

## ❌ NOT COMPLETED (Ready to Run)

### Priority 1: B0 + B2 Ensemble (Multi-Scale)
- **Script**: `evaluate_ensemble_general.py`
- **Models**: 
  - Model 1: `ablation_full_enhanced_b0_224_jitter`
  - Model 2: `ablation_full_enhanced_b2`
- **Method**: `weighted`
- **Status**: ❌ Not started

### Priority 2: B0 + B1 + B2 Ensemble (3-way)
- **Script**: `evaluate_ensemble_multiway.py`
- **Models**: 
  - `ablation_full_enhanced_b0_224_jitter`
  - `ablation_full_enhanced_b1`
  - `ablation_full_enhanced_b2`
- **Method**: `majority`
- **Status**: ❌ Not started

### Priority 3: B0 + B1 + B2 + Ordinal Ensemble (4-way)
- **Script**: `evaluate_ensemble_multiway.py`
- **Models**: 
  - `ablation_full_enhanced_b0_224_jitter`
  - `ablation_full_enhanced_b1`
  - `ablation_full_enhanced_b2`
  - `ordinal_b0_224_jitter_weighted`
- **Method**: `majority`
- **Status**: ❌ Not started

### Priority 4: B0 + B1 + B2 + B3 + Ordinal Ensemble (5-way)
- **Script**: `evaluate_ensemble_multiway.py`
- **Models**: 
  - `ablation_full_enhanced_b0_224_jitter`
  - `ablation_full_enhanced_b1`
  - `ablation_full_enhanced_b2`
  - `ablation_full_enhanced_b3`
  - `ordinal_b0_224_jitter_weighted`
- **Method**: `majority`
- **Status**: ❌ Not started

---

## Notes

- All experiments should run on **validation set first** (`--split val`)
- After reviewing val results, run best combinations on test set
- Expected improvement: 0.1-1.0% depending on number of models

