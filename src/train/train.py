"""Training script for BCS classification with transfer learning support."""
import argparse
import os
import shutil
import time
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import traceback
from torch.utils.data import DataLoader

from .dataset import BcsDataset
from .early_stopping import EarlyStopping
from ..models import create_model
from ..models.losses import create_ordinal_loss
from ..eval import evaluate, plot_confusion_matrix


def get_device(cfg_device=None):
    """Get appropriate device (cpu/cuda/mps) based on config and availability."""
    if cfg_device in {"cpu", "cuda", "mps"}:
        return cfg_device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_optimizer(model, optimizer_name: str, lr: float, weight_decay: float = 1e-4):
    """Create optimizer based on config."""
    if optimizer_name.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, scheduler_name: str, epochs: int):
    """Create learning rate scheduler based on config."""
    if scheduler_name is None or scheduler_name == "null":
        return None
    elif scheduler_name.lower() == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name.lower() == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def main(cfg_path: str):
    """Main training function."""
    try:
        # ---------- Load config ----------
        print(f"Loading config from: {cfg_path}")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        if cfg is None:
            raise ValueError(f"Config file is empty or invalid: {cfg_path}")
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        traceback.print_exc()
        raise

    # ---------- Read settings ----------
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})

    train_csv = data_cfg.get("train", "data/train.csv")
    val_csv = data_cfg.get("val", "data/val.csv")
    img_size = int(data_cfg.get("img_size", 224))
    do_aug = bool(data_cfg.get("do_aug", False))

    backbone = model_cfg.get("backbone", "efficientnet_b0")
    num_classes = int(model_cfg.get("num_classes", 5))
    pretrained = bool(model_cfg.get("pretrained", True))
    finetune_mode = model_cfg.get("finetune_mode", "full")
    head_type = model_cfg.get("head_type", "classification")  # "classification" or "ordinal"

    bs = int(train_cfg.get("batch_size", 32))
    epochs = int(train_cfg.get("epochs", 20))
    lr = float(train_cfg.get("lr", 3e-4))
    seed = int(train_cfg.get("seed", 42))
    num_workers = int(train_cfg.get("num_workers", 2))
    device = get_device(train_cfg.get("device"))
    
    optimizer_name = train_cfg.get("optimizer", "adamw")
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    scheduler_name = train_cfg.get("scheduler")
    label_smoothing = float(train_cfg.get("label_smoothing", 0.05))
    
    # Early stopping config
    es_cfg = train_cfg.get("early_stopping", {})
    es_enabled = bool(es_cfg.get("enabled", False))
    es_patience = int(es_cfg.get("patience", 5))
    es_monitor = es_cfg.get("monitor", "val_macro_f1")
    es_mode = es_cfg.get("mode", "max")

    # Output directory
    exp_name = cfg.get("exp_name", f"exp_{finetune_mode}_{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    out_dir = cfg.get("out_dir", f"outputs/{exp_name}")
    os.makedirs(out_dir, exist_ok=True)

    # Print config summary
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Config: {cfg_path}")
    print(f"Device: {device} | Batch size: {bs} | Epochs: {epochs}")
    print(f"Backbone: {backbone} | Finetune mode: {finetune_mode} | Head type: {head_type}")
    print(f"Pretrained: {pretrained} | Learning rate: {lr}")
    print(f"Early stopping: {es_enabled} (patience={es_patience}, monitor={es_monitor})")
    print(f"Output directory: {out_dir}")
    print("=" * 60)

    set_seed(seed)

    # ---------- Data ----------
    print("\nBuilding datasets...")
    try:
        if not os.path.exists(train_csv):
            raise FileNotFoundError(f"Train CSV not found: {train_csv}")
        if not os.path.exists(val_csv):
            raise FileNotFoundError(f"Val CSV not found: {val_csv}")
        
        print(f"  Loading train dataset from: {train_csv}")
        ds_tr = BcsDataset(train_csv, img_size=img_size, train=True, do_aug=do_aug)
        print(f"  Loading val dataset from: {val_csv}")
        ds_va = BcsDataset(val_csv, img_size=img_size, train=False, do_aug=False)
    except Exception as e:
        print(f"ERROR: Failed to load datasets: {e}")
        traceback.print_exc()
        raise
    
    dl_tr = DataLoader(
        ds_tr,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=train_cfg.get("persistent_workers", False),
        prefetch_factor=train_cfg.get("prefetch_factor", 2) if num_workers > 0 else None
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=train_cfg.get("persistent_workers", False),
        prefetch_factor=train_cfg.get("prefetch_factor", 2) if num_workers > 0 else None
    )
    print(f"Train samples: {len(ds_tr)} | Val samples: {len(ds_va)}")

    # ---------- Model ----------
    print("\nCreating model...")
    try:
        print(f"  Backbone: {backbone}, Head type: {head_type}, Classes: {num_classes}")
        model = create_model(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            finetune_mode=finetune_mode,
            head_type=head_type
        )
        print(f"  Moving model to device: {device}")
        model.to(device)
        print(f"  ✓ Model created successfully")
    except Exception as e:
        print(f"ERROR: Failed to create model: {e}")
        traceback.print_exc()
        raise

    # ---------- Optimizer / Loss / Scheduler ----------
    try:
        if head_type == "ordinal":
            # Ordinal regression loss
            print("  Creating ordinal loss...")
            threshold_weights = train_cfg.get("ordinal_threshold_weights", None)
            if threshold_weights:
                print(f"    Using threshold weights: {threshold_weights}")
            criterion = create_ordinal_loss(num_classes, threshold_weights=threshold_weights)
            print("  ✓ Ordinal loss created")
        else:
            # Standard classification loss
            print("  Creating classification loss...")
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            print("  ✓ Classification loss created")
    except Exception as e:
        print(f"ERROR: Failed to create loss function: {e}")
        traceback.print_exc()
        raise
    
    optimizer = get_optimizer(model, optimizer_name, lr, weight_decay)
    scheduler = get_scheduler(optimizer, scheduler_name, epochs)

    # ---------- Early Stopping ----------
    early_stopping = None
    if es_enabled:
        early_stopping = EarlyStopping(
            patience=es_patience,
            monitor=es_monitor,
            mode=es_mode
        )

    # ---------- Training Loop ----------
    print("\nStarting training...")
    print("-" * 60)
    
    best_metrics = {
        "acc": 0.0,
        "macro_f1": 0.0,
        "underweight_recall": 0.0
    }
    class_names = eval_cfg.get("class_names", ["3.25", "3.5", "3.75", "4.0", "4.25"])

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        num_steps = 0
        
        for step, (xb, yb) in enumerate(dl_tr, start=1):
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_steps += 1
            
            if step % 50 == 0:
                print(f"  Epoch {epoch:02d} | Step {step:4d} | Loss: {loss.item():.4f}")

        train_loss = running_loss / max(1, num_steps)

        # Validation phase
        eval_results = evaluate(model, dl_va, device, class_names=class_names, head_type=head_type)
        
        # Get monitored metric for early stopping
        if es_monitor == "val_acc":
            monitor_score = eval_results["acc"]
        elif es_monitor == "val_macro_f1":
            monitor_score = eval_results["macro_f1"]
        elif es_monitor == "val_underweight_recall":
            monitor_score = eval_results["underweight_recall"]
        else:
            monitor_score = eval_results["acc"]  # Default fallback

        # Print epoch results
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
              f"Val Acc: {eval_results['acc']:.4f} | "
              f"Val Macro-F1: {eval_results['macro_f1']:.4f} | "
              f"Underweight Recall: {eval_results['underweight_recall']:.4f}")

        # Update best metrics and save checkpoint
        if eval_results["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = eval_results.copy()
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
            print(f"  -> New best model saved (Macro-F1: {best_metrics['macro_f1']:.4f})")

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            if epoch % 5 == 0:
                print(f"  Learning rate: {current_lr:.6f}")

        # Early stopping check
        if early_stopping is not None:
            if early_stopping(monitor_score):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best {es_monitor}: {early_stopping.best_score:.4f}")
                break

        print("-" * 60)

    # ---------- Final Evaluation and Saving ----------
    print("\nFinal evaluation on best model...")
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt")))
    final_eval = evaluate(model, dl_va, device, class_names=class_names, head_type=head_type)
    
    # Save confusion matrix
    if eval_cfg.get("save_confusion_matrix", True):
        cm_path = os.path.join(out_dir, "confusion_matrix.png")
        plot_confusion_matrix(
            final_eval["confusion_matrix"],
            class_names,
            save_path=cm_path
        )

    # Save metrics to JSON
    import json
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final_eval, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Save config for reproducibility
    shutil.copy2(cfg_path, os.path.join(out_dir, "config.yaml"))

    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best Val Accuracy: {best_metrics['acc']:.4f}")
    print(f"Best Val Macro-F1: {best_metrics['macro_f1']:.4f}")
    print(f"Best Underweight Recall: {best_metrics['underweight_recall']:.4f}")
    print(f"\nArtifacts saved to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BCS classification model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML file")
    args = parser.parse_args()
    main(args.config)
