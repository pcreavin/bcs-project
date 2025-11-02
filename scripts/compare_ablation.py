"""Compare results from ablation experiments."""
import json
import pathlib
import pandas as pd
from typing import Dict, List
import argparse


def find_experiment_dirs(base_dir: str = "outputs") -> Dict[str, pathlib.Path]:
    """Find all ablation experiment directories."""
    base = pathlib.Path(base_dir)
    experiments = {}
    
    for exp_dir in base.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # Look for metrics.json
        metrics_file = exp_dir / "metrics.json"
        if metrics_file.exists():
            # Try to infer experiment type from directory name or config
            config_file = exp_dir / "config.yaml"
            if config_file.exists():
                import yaml
                with open(config_file) as f:
                    cfg = yaml.safe_load(f)
                    exp_name = cfg.get("exp_name") or cfg.get("model", {}).get("finetune_mode", "unknown")
                    experiments[exp_name] = exp_dir
            else:
                experiments[exp_dir.name] = exp_dir
    
    return experiments


def load_metrics(exp_dir: pathlib.Path) -> Dict:
    """Load metrics from an experiment directory."""
    metrics_file = exp_dir / "metrics.json"
    if not metrics_file.exists():
        return None
    
    with open(metrics_file) as f:
        return json.load(f)


def create_comparison_table(experiments: Dict[str, pathlib.Path]) -> pd.DataFrame:
    """Create a comparison table from experiment results."""
    rows = []
    
    # Preferred order
    order = ["scratch", "head_only", "last_block", "full"]
    
    for exp_name in order:
        if exp_name not in experiments:
            continue
        
        metrics = load_metrics(experiments[exp_name])
        if metrics is None:
            continue
        
        rows.append({
            "Experiment": exp_name,
            "Accuracy": f"{metrics['acc']:.4f}",
            "Macro-F1": f"{metrics['macro_f1']:.4f}",
            "Weighted-F1": f"{metrics['weighted_f1']:.4f}",
            "Underweight Recall": f"{metrics['underweight_recall']:.4f}",
        })
        
        # Add per-class recall
        for i, (class_name, recall) in enumerate(zip(metrics['class_names'], metrics['per_class_recall'])):
            rows[-1][f"Recall {class_name}"] = f"{recall:.4f}"
    
    # Add any experiments not in preferred order
    for exp_name, exp_dir in experiments.items():
        if exp_name not in order:
            metrics = load_metrics(exp_dir)
            if metrics:
                rows.append({
                    "Experiment": exp_name,
                    "Accuracy": f"{metrics['acc']:.4f}",
                    "Macro-F1": f"{metrics['macro_f1']:.4f}",
                    "Weighted-F1": f"{metrics['weighted_f1']:.4f}",
                    "Underweight Recall": f"{metrics['underweight_recall']:.4f}",
                })
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Compare ablation experiment results")
    parser.add_argument("--output", type=str, default="ablation_comparison.csv",
                        help="Output CSV file path")
    parser.add_argument("--outputs-dir", type=str, default="outputs",
                        help="Directory containing experiment outputs")
    args = parser.parse_args()
    
    print("Finding experiments...")
    experiments = find_experiment_dirs(args.outputs_dir)
    
    if not experiments:
        print(f"No experiments found in {args.outputs_dir}")
        return
    
    print(f"Found {len(experiments)} experiments:")
    for name, path in experiments.items():
        print(f"  - {name}: {path}")
    
    print("\nLoading metrics...")
    comparison_df = create_comparison_table(experiments)
    
    if comparison_df.empty:
        print("No metrics found in experiments.")
        return
    
    # Save to CSV
    comparison_df.to_csv(args.output, index=False)
    print(f"\nComparison saved to {args.output}")
    
    # Print formatted table
    print("\n" + "=" * 80)
    print("ABLATION COMPARISON RESULTS")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)
    
    # Highlight best results
    print("\nBest Results:")
    numeric_cols = ["Accuracy", "Macro-F1", "Underweight Recall"]
    for col in numeric_cols:
        if col in comparison_df.columns:
            # Convert to float for comparison
            values = comparison_df[col].astype(str).str.replace("[^0-9.]", "", regex=True).astype(float)
            best_idx = values.idxmax()
            best_exp = comparison_df.iloc[best_idx]["Experiment"]
            best_val = comparison_df.iloc[best_idx][col]
            print(f"  {col}: {best_val} ({best_exp})")


if __name__ == "__main__":
    main()

