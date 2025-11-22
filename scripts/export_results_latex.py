#!/usr/bin/env python3
"""Export experiment results to LaTeX tables and figures for milestone/final report."""
import argparse
import json
import pathlib
import pandas as pd
from typing import Dict, List
import yaml


def load_metrics(exp_dir: pathlib.Path) -> Dict:
    """Load metrics from an experiment directory."""
    metrics_file = exp_dir / "metrics.json"
    if not metrics_file.exists():
        return None
    with open(metrics_file) as f:
        return json.load(f)


def create_results_table(outputs_dir: str = "outputs") -> pd.DataFrame:
    """Create comprehensive results table from all experiments."""
    base = pathlib.Path(outputs_dir)
    rows = []
    
    # Order of experiments
    order = ["scratch", "head_only", "last_block", "full"]
    
    for exp_name in order:
        exp_dir = None
        # Find experiment directory
        for d in base.iterdir():
            if not d.is_dir():
                continue
            config_file = d / "config.yaml"
            if config_file.exists():
                with open(config_file) as f:
                    cfg = yaml.safe_load(f)
                    if cfg.get("exp_name") == exp_name or cfg.get("model", {}).get("finetune_mode") == exp_name:
                        exp_dir = d
                        break
        
        if exp_dir is None:
            continue
        
        metrics = load_metrics(exp_dir)
        if metrics is None:
            continue
        
        # Format experiment name for display
        display_name = {
            "scratch": "From Scratch",
            "head_only": "Head Only",
            "last_block": "Last Block",
            "full": "Full Fine-tuning"
        }.get(exp_name, exp_name.replace("_", " ").title())
        
        rows.append({
            "Method": display_name,
            "Accuracy": metrics['acc'],
            "Macro-F1": metrics['macro_f1'],
            "Underweight Recall": metrics['underweight_recall'],
        })
    
    return pd.DataFrame(rows)


def export_table_latex(df: pd.DataFrame, output_path: str, caption: str, label: str):
    """Export DataFrame to LaTeX table with nice formatting."""
    # Convert numeric columns to strings with 4 decimal places
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].apply(lambda x: f"{x:.4f}")
    
    latex_code = df.to_latex(
        index=False,
        escape=False,
        column_format="l" + "c" * (len(df.columns) - 1),
        caption=caption,
        label=label
    )
    
    # Wrap in table environment
    full_tex = f"""\\begin{{table}}[htbp]
\\centering
{latex_code}
\\end{{table}}
"""
    
    with open(output_path, "w") as f:
        f.write(full_tex)
    
    print(f"LaTeX table saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export results to LaTeX format")
    parser.add_argument("--outputs-dir", type=str, default="outputs",
                        help="Directory containing experiment outputs")
    parser.add_argument("--output-dir", type=str, default="docs/latex",
                        help="Directory to save LaTeX files")
    args = parser.parse_args()
    
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading experiment results...")
    df = create_results_table(args.outputs_dir)
    
    if df.empty:
        print("No experiments found!")
        return
    
    print("\nResults table:")
    print(df.to_string(index=False))
    
    # Export main comparison table
    export_table_latex(
        df,
        output_dir / "results_table.tex",
        caption="Comparison of Transfer Learning Strategies on BCS Classification",
        label="tab:transfer_learning_comparison"
    )
    
    # Export detailed metrics table
    # (You can extend this to include per-class metrics)
    
    print(f"\nLaTeX files saved to {output_dir}/")


if __name__ == "__main__":
    main()












