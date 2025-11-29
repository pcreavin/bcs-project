# Experiment Summary

## 1. Transfer Learning Ablation Study

**Objective**: Compare different transfer learning strategies to determine optimal fine-tuning approach.

### 1.1 Scratch (No Pretraining)
- **Config**: `configs/ablation_scratch.yaml`
- **Model**: EfficientNet-B0, pretrained=False
- **Data**: Preprocessed 224px images
- **Status**: ✅ Complete
- **Results**: 
  - Accuracy: 69.83%
  - Macro-F1: 70.36%
  - Underweight Recall: 67.17%
  - See `outputs/ablation_scratch/metrics.json`
- **Purpose**: Baseline without pretrained weights to measure benefit of transfer learning

### 1.2 Head Only
- **Config**: `configs/ablation_head_only.yaml`
- **Model**: EfficientNet-B0, freeze backbone, train only classifier
- **Data**: Preprocessed 224px images
- **Status**: ✅ Complete
- **Results**: 
  - Accuracy: 40.73%
  - Macro-F1: 40.99%
  - Underweight Recall: 43.63%
  - See `outputs/ablation_head_only/metrics.json`
- **Purpose**: Test if pretrained features alone are sufficient

### 1.3 Last Block
- **Config**: `configs/ablation_last_block.yaml`
- **Model**: EfficientNet-B0, freeze early layers, unfreeze last block(s)
- **Data**: Preprocessed 224px images
- **Status**: ✅ Complete
- **Results**: 
  - Accuracy: 72.17%
  - Macro-F1: 72.80%
  - Underweight Recall: 73.01%
  - See `outputs/ablation_last_block/metrics.json`
- **Purpose**: Balance between feature reuse and task-specific adaptation

### 1.4 Full Fine-tuning
- **Config**: `configs/ablation_full.yaml`
- **Model**: EfficientNet-B0, fine-tune all parameters
- **Data**: Preprocessed 224px images
- **Status**: ✅ Complete
- **Results**: 
  - Accuracy: 92.26%
  - Macro-F1: 92.53%
  - Underweight Recall: 91.59%
  - See `outputs/ablation_full/metrics.json`
- **Purpose**: Maximum adaptation to target task

---

## 2. Enhanced Configuration Experiments

**Objective**: Improve baseline performance with better training strategies.

### 2.1 Enhanced Baseline
- **Config**: `configs/ablation_full_enhanced.yaml`
- **Model**: EfficientNet-B0, full fine-tuning
- **Data**: Preprocessed 224px images
- **Enhancements**: Data augmentation (horizontal flip + rotation), cosine scheduler
- **Status**: ✅ Complete
- **Results**: 
  - Accuracy: 96.13%
  - Macro-F1: 96.21%
  - Underweight Recall: 95.13%
  - See `outputs/ablation_full_enhanced/metrics.json`

### 2.2 ROI Cropping with Jitter
- **Config**: `configs/ablation_full_enhanced_b0_224_jitter.yaml` (via outputs)
- **Model**: EfficientNet-B0, full fine-tuning
- **Data**: Original images with ROI cropping (224px), augmentation enabled
- **Enhancements**: ROI cropping instead of preprocessed images, crop jitter
- **Status**: ✅ Complete
- **Results**: 
  - Accuracy: 96.25%
  - Macro-F1: 96.37%
  - Underweight Recall: 94.51%
  - See `outputs/ablation_full_enhanced_b0_224_jitter/metrics.json`
- **Purpose**: Test if ROI cropping improves focus on relevant features

### 2.3 Larger Image Size
- **Config**: `configs/ablation_full_enhanced_b0_320.yaml` (via outputs)
- **Model**: EfficientNet-B0, full fine-tuning
- **Data**: 320px images
- **Status**: ✅ Complete
- **Results**: See `outputs/ablation_full_enhanced_b0_320/metrics.json`
- **Purpose**: Test impact of higher resolution

### 2.4 Larger Model Variants
- **Models**: EfficientNet-B1, B2, B3
- **Configs**: `configs/ablation_full_enhanced_b1.yaml`, `b2.yaml`, `b3.yaml`
- **Status**: ✅ Complete
- **Results**: 
  - B2: 94.93% accuracy, 95.09% Macro-F1 (See `outputs/ablation_full_enhanced_b2/metrics.json`)
  - B1, B3: See respective output directories
- **Purpose**: Test if larger models improve performance

---

## 3. Ordinal Regression Experiments

**Objective**: Test CORAL-style ordinal regression for better handling of ordered classes.

### 3.1 Ordinal Baseline (Unweighted)
- **Config**: `configs/ordinal_b0_224_jitter.yaml` (via outputs)
- **Model**: EfficientNet-B0 with ordinal head (4 threshold logits)
- **Data**: ROI cropping, augmentation enabled
- **Decoding**: threshold_count
- **Status**: ✅ Complete
- **Results**: See `outputs/ordinal_b0_224_jitter/metrics.json`

### 3.2 Ordinal with Threshold Weighting
- **Config**: `configs/ordinal_b0_224_jitter_weighted.yaml` (via outputs)
- **Model**: EfficientNet-B0 with ordinal head
- **Threshold Weights**: [2.0, 1.0, 1.0, 1.0] (weight first threshold to prioritize underweight class)
- **Data**: ROI cropping, augmentation enabled
- **Status**: ✅ Complete
- **Results**:
  - Accuracy: 94.16%
  - Macro-F1: 94.53%
  - Underweight Recall: 93.10%
  - MAE (BCS): 0.0168
  - Ordinal Accuracy (±1 class): 99.18%
  - See `outputs/ordinal_b0_224_jitter_weighted/metrics.json`

### 3.3 Ordinal Decoding Method Comparison
- **Experiment**: Compare different decoding methods on trained ordinal model
- **Methods Tested**:
  - `threshold_count`: Count thresholds >= 0.5 (baseline)
  - `expected_value`: Use expected value of thresholds
  - `max_prob`: Use threshold with highest probability
- **Status**: ✅ Complete
- **Results**: 
  - `threshold_count`: 94.16% accuracy (best)
  - `expected_value`: 93.98% accuracy
  - `max_prob`: 36.49% accuracy (not suitable for ordinal regression)
  - See `outputs/ordinal_b0_224_jitter_weighted/decoding_comparison_val.json`
- **Conclusion**: `threshold_count` is the best decoding method

---

## 4. Ensemble Experiments

**Objective**: Combine classification baseline and ordinal regression models for improved performance.

### 4.1 Ensemble: Classification + Ordinal
- **Baseline Model**: `outputs/ablation_full_enhanced_b0_224_jitter/best_model.pt`
- **Ordinal Model**: `outputs/ordinal_b0_224_jitter_weighted/best_model.pt`
- **Ensemble Rule**:
  - Same class → use it
  - Differ by ±1 class → use baseline
  - Differ by >1 class → use ordinal
- **Decoding Method**: threshold_count
- **Status**: ✅ Complete (validation and test sets)
- **Results**:
  - **Validation Set**:
    - Accuracy: 96.09%
    - Macro-F1: 96.22%
    - Underweight Recall: 94.07%
    - MAE (BCS): 0.0120
  - **Test Set**:
    - Accuracy: 96.25%
    - Macro-F1: 96.41%
    - Underweight Recall: 95.58%
    - MAE (BCS): 0.0118
  - Agreement: 93.7% same predictions, 5.4% differ by ±1, 0.9% differ by >1
  - See `outputs/ensemble_ablation_full_enhanced_b0_224_jitter_ordinal_b0_224_jitter_weighted_threshold_count/ensemble_metrics_val.json` and `ensemble_metrics_test.json`
- **Purpose**: Leverage strengths of both approaches

---

## 5. ROI Robustness Experiments

**Objective**: Test model robustness to background context by comparing ROI-only vs ROI+context (address reviewer feedback).

### 5.1 ROI-Only (0% Padding)
- **Config**: `configs/roi_robustness_roi_only.yaml`
- **Crop Padding**: 0.0 (tight crop to bounding box)
- **Status**: ⏳ In Progress
- **Purpose**: Baseline with no background context

### 5.2 ROI + 5% Context
- **Config**: `configs/roi_robustness_padding_05.yaml`
- **Crop Padding**: 0.05 (5% margin on each side)
- **Status**: ⏳ In Progress
- **Purpose**: Test with minimal background context

### 5.3 ROI + 10% Context
- **Config**: `configs/roi_robustness_padding_10.yaml`
- **Crop Padding**: 0.10 (10% margin on each side)
- **Status**: ⏳ In Progress
- **Purpose**: Test with more background context

**Expected Outcome**: If model is robust, all three should have similar accuracy. Significant differences indicate reliance on background cues.

---

## 6. Alternative Architecture Experiments (Ishan)

**Objective**: Test modern architectures that may outperform EfficientNet-B0.

### 6.1 ConvNeXt-Tiny @ 224px
- **Architecture**: ConvNeXt-Tiny
- **Input Size**: 224px
- **Rationale**: Modern convnet with LayerNorm + large kernels; often outperforms EfficientNet-B0 on fine-grained tasks while staying lightweight
- **Status**: ⏳ Pending

### 6.2 Swin-T Transformer @ 224px
- **Architecture**: Swin-T (Vision Transformer)
- **Input Size**: 224px
- **Rationale**: Hierarchical Vision Transformer; strong at capturing long-range context, worth testing if compute allows
- **Status**: ⏳ Pending

### 6.3 RegNetY-8GF @ 224px
- **Architecture**: RegNetY-8GF
- **Input Size**: 224px
- **Rationale**: Balanced accuracy/efficiency, includes SE-style channel attention; good contrast to EfficientNet and often robust
- **Status**: ⏳ Pending

---

## 7. Ensemble Learning Methods (Quinten)

**Objective**: Explore alternative ensemble approaches beyond the current override rule.

### 7.1 Additional Ensemble Methods
- **Status**: ⏳ Pending
- **Methods to Test**: TBD
- **Purpose**: Test if different ensemble strategies (e.g., weighted voting, stacking) improve performance

---

## 8. Test Set Analysis & Grad-CAM

**Objective**: Analyze model behavior on held-out test set and generate explainability visualizations.

### 8.1 Test Set Evaluation
- **Status**: ⏳ Pending
- **Purpose**: Final evaluation on held-out test set for all best models

### 8.2 Grad-CAM Visualizations
- **Method**: Gradient-weighted Class Activation Mapping
- **Status**: ⏳ Pending
- **Purpose**: 
  - Visualize which image regions the model focuses on
  - Verify model is looking at relevant anatomical features (tailhead/rump area)
  - Identify potential failure cases
  - Generate explainability for final report

---

## Summary Statistics

### Best Single Models (Validation Set)
- **Classification Baseline**: 96.25% accuracy (EfficientNet-B0 with ROI cropping)
- **Ordinal Regression**: 94.16% accuracy (with threshold weighting, 99.18% ordinal accuracy ±1 class)
- **Ensemble**: 96.25% accuracy on test set (96.09% on validation)

### Key Findings
1. Full fine-tuning outperforms other transfer learning strategies (92.26% vs 72.17% last_block, 40.73% head_only, 69.83% scratch)
2. ROI cropping with augmentation improves over preprocessed images (96.25% vs 96.13%)
3. Ordinal regression provides excellent ordinal accuracy (±1 class: 99.18%) with lower exact accuracy (94.16%)
4. Ensemble approach combines strengths of both classification and ordinal methods

---

## Next Steps

1. ⏳ Complete ROI robustness experiments (0%, 5%, 10% padding)
2. ⏳ Run Ishan's alternative architecture experiments
3. ⏳ Implement Quinten's additional ensemble methods
4. ⏳ Run final test set evaluation
5. ⏳ Generate Grad-CAM visualizations on test set
6. ⏳ Final model selection and report preparation


