"""Baseline models utilities for Person A.

Provides functions to extract frozen features from a timm model and train
classical ML baselines (Logistic Regression and SVM) on those features.

Expected usage:
  model = create_model(backbone, num_classes, pretrained=True, finetune_mode='head_only')
  model.to(device)
  X_train, y_train = extract_frozen_features(model, train_loader, device)
  X_val, y_val = extract_frozen_features(model, val_loader, device)
  train_logistic_regression(X_train, y_train, X_val, y_val, out_dir)
  train_svm(X_train, y_train, X_val, y_val, out_dir)
"""

from typing import Tuple, Optional
import os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib


def extract_frozen_features(model: torch.nn.Module, dataloader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and labels from a frozen pretrained model.

    The function prefers using `model.forward_features(x)` (timm models). If that
    attribute isn't available it will try passing inputs through the model and
    taking the penultimate activation by removing the last classifier if possible.

    Returns:
        X: ndarray shape (N, D)
        y: ndarray shape (N,)
    """
    model.eval()
    features = []
    labels = []

    # Helper to get features from a batch
    def _get_feat(x: torch.Tensor) -> torch.Tensor:
        # Try timm-style forward_features
        if hasattr(model, "forward_features"):
            return model.forward_features(x)

        # Try common attribute names for backbone without classifier
        # This is a best-effort fallback; not all models support it.
        for attr in ("features", "backbone", "stem"):
            if hasattr(model, attr):
                backbone = getattr(model, attr)
                try:
                    return backbone(x)
                except Exception:
                    break

        # Last resort: run full forward and try to remove classifier layer output
        out = model(x)
        # If output is logits, try to use flatten features from previous modules is not possible here;
        # simply return the output (may be logits) to allow training baselines on them.
        return out

    with torch.no_grad():
        for batch in dataloader:
            # Support dataloaders returning (x,y) or dicts
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            elif isinstance(batch, dict):
                x, y = batch["image"], batch.get("label")
            else:
                raise ValueError("Unsupported batch format from dataloader. Expected (x,y) or dict with 'image'/'label'.")

            x = x.to(device)
            feat = _get_feat(x)

            # Move to CPU and convert to numpy
            feat_np = feat.detach().cpu().numpy()
            # If features have spatial dims, apply global average pooling
            if feat_np.ndim > 2:
                # Global average pool: (batch, channels, H, W) -> (batch, channels)
                feat_np = feat_np.mean(axis=tuple(range(2, feat_np.ndim)))

            features.append(feat_np)
            labels.append(y.cpu().numpy() if torch.is_tensor(y) else np.array(y))

    X = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y


def _fit_and_eval_pipeline(pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray, out_dir: Optional[str], name: str):
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "macro_f1": float(f1_score(y_val, y_pred, average="macro")),
    }

    report = classification_report(y_val, y_pred, output_dict=True)
    cm = confusion_matrix(y_val, y_pred)

    # Save model and artifacts if requested
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, f"{name}_model.joblib")
        metrics_path = os.path.join(out_dir, f"{name}_metrics.npy")
        report_path = os.path.join(out_dir, f"{name}_report.joblib")
        cm_path = os.path.join(out_dir, f"{name}_confusion.npy")

        joblib.dump(pipeline, model_path)
        np.save(metrics_path, metrics)
        joblib.dump(report, report_path)
        np.save(cm_path, cm)

    return metrics, report, cm


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray,
                              out_dir: Optional[str] = None,
                              C: float = 1.0) -> Tuple[dict, dict, np.ndarray]:
    """Train a Logistic Regression baseline with standard scaling.

    Returns (metrics, report, confusion_matrix)
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000, multi_class="multinomial", solver="lbfgs")),
    ])

    return _fit_and_eval_pipeline(pipe, X_train, y_train, X_val, y_val, out_dir, "logreg")


def train_svm(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              out_dir: Optional[str] = None,
              kernel: str = "rbf", C: float = 1.0) -> Tuple[dict, dict, np.ndarray]:
    """Train an SVM baseline (with standard scaling).

    For multi-class problems, SVC uses one-vs-one by default. Setting probability=True
    enables probability estimates (slower).
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel=kernel, C=C, probability=True)),
    ])

    return _fit_and_eval_pipeline(pipe, X_train, y_train, X_val, y_val, out_dir, f"svm_{kernel}")


__all__ = [
    "extract_frozen_features",
    "train_logistic_regression",
    "train_svm",
]
