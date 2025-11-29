# 5-Way Ensemble Model Mapping

**Experiment**: `ensemble_5way_ablation_full_enhanced_b0_224_jitter_ablation_full_enhanced_b1_ablation_full_enhanced_b2_and2more_majority`

**Method**: Majority voting across 5 models

**Evaluation Split**: Validation set (8,035 samples)

---

## Model Identification

### Model 1: EfficientNet-B0 @ 224px with ROI Cropping + Jitter
- **Type**: Classification
- **Checkpoint**: `outputs/ablation_full_enhanced_b0_224_jitter/best_model.pt`
- **Config**: `outputs/ablation_full_enhanced_b0_224_jitter/config.yaml`
- **Performance**:
  - Accuracy: **96.25%** ✅ (Best individual model)
  - Macro-F1: 96.37%
  - Underweight Recall: 94.51%

### Model 2: EfficientNet-B1 @ 320px
- **Type**: Classification
- **Checkpoint**: `outputs/ablation_full_enhanced_b1/ablation_full_enhanced_b1/best_model.pt`
- **Config**: `outputs/ablation_full_enhanced_b1/ablation_full_enhanced_b1/config.yaml`
- **Performance**:
  - Accuracy: **68.34%** ⚠️ (Weakest individual model)
  - Macro-F1: 69.07%
  - Underweight Recall: 74.42%

### Model 3: EfficientNet-B2
- **Type**: Classification
- **Checkpoint**: `outputs/ablation_full_enhanced_b2/best_model.pt`
- **Config**: `outputs/ablation_full_enhanced_b2/config.yaml`
- **Performance**:
  - Accuracy: **81.14%** ⚠️
  - Macro-F1: 81.61%
  - Underweight Recall: 90.44%

### Model 4: EfficientNet-B3
- **Type**: Classification
- **Checkpoint**: `outputs/ablation_full_enhanced_b3/best_model.pt`
- **Config**: `outputs/ablation_full_enhanced_b3/config.yaml`
- **Performance**:
  - Accuracy: **84.39%** ⚠️
  - Macro-F1: 84.74%
  - Underweight Recall: 78.58%

### Model 5: EfficientNet-B0 with Ordinal Head
- **Type**: Ordinal Regression
- **Checkpoint**: `outputs/ordinal_b0_224_jitter_weighted/best_model.pt`
- **Config**: `outputs/ordinal_b0_224_jitter_weighted/config.yaml`
- **Decoding Method**: `threshold_count`
- **Performance**:
  - Accuracy: **27.90%** ❌ (Severely underperforming - likely loading/decoding issue)
  - Macro-F1: 19.69%
  - Underweight Recall: 0.62% (essentially failed)

**Note**: Model 5's performance is abnormally low. When properly loaded, this model should achieve ~94% accuracy. This suggests the ordinal model was not loaded correctly during ensemble evaluation, or the decoding method failed.

---

## Ensemble Results

### Final Ensemble (Majority Vote)
- **Accuracy**: **93.73%**
- **Macro-F1**: 93.96%
- **Weighted-F1**: 93.74%
- **Underweight Recall**: 92.92%

### Agreement Statistics
- **Total Samples**: 8,035
- **All 5 models agree**: 1,187 samples (14.77%)
- **Agreement is low**, indicating high disagreement between models

---

## Performance Comparison

| Model | Accuracy | Macro-F1 | Underweight Recall | Status |
|-------|----------|----------|-------------------|--------|
| Model 1 (B0) | 96.25% | 96.37% | 94.51% | ✅ Best |
| Model 2 (B1) | 68.34% | 69.07% | 74.42% | ⚠️ Weak |
| Model 3 (B2) | 81.14% | 81.61% | 90.44% | ⚠️ Moderate |
| Model 4 (B3) | 84.39% | 84.74% | 78.58% | ⚠️ Moderate |
| Model 5 (Ordinal) | 27.90% | 19.69% | 0.62% | ❌ Failed |
| **Ensemble** | **93.73%** | **93.96%** | **92.92%** | ⚠️ Worse than Model 1 |

---

## Key Insights

1. **Best Individual Model**: Model 1 (B0) outperforms the ensemble (96.25% vs 93.73%)
2. **Ensemble Underperformance**: The ensemble is worse than the best model because:
   - Model 2 (B1) and Model 5 (Ordinal) are very weak
   - Low agreement (14.77%) means majority voting is noisy
   - Weak models are dragging down the strong model
3. **Ordinal Model Issue**: Model 5's 27.90% accuracy indicates a serious problem - likely:
   - Incorrect model loading
   - Wrong decoding method
   - State dict mismatch
4. **Recommendation**: 
   - Use Model 1 (B0) alone (96.25%) instead of this ensemble
   - Fix Model 5's loading/decoding issue if ordinal ensemble is desired
   - Consider removing weak models (B1, B2, B3) from ensemble

---

## File Structure

Each model's individual results are saved as:
- `confusion_matrix_model1_classification_val.png` → Model 1 (B0)
- `confusion_matrix_model2_classification_val.png` → Model 2 (B1)
- `confusion_matrix_model3_classification_val.png` → Model 3 (B2)
- `confusion_matrix_model4_classification_val.png` → Model 4 (B3)
- `confusion_matrix_model5_ordinal_val.png` → Model 5 (Ordinal)
- `confusion_matrix_ensemble_val.png` → Final ensemble predictions

Full metrics are in: `ensemble_metrics_val.json`

