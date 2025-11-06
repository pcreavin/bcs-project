# LaTeX Export Guide

This guide explains how to export experiment results to LaTeX format for your milestone/final report.

## Setup

**Prerequisites:**
- ✅ MacTeX installed (you have it: `/usr/local/bin/pdflatex`)
- ✅ pandas installed (already in requirements.txt)
- ✅ Python packages installed

**No additional packages needed!** pandas has built-in LaTeX export via `to_latex()`.

## Quick Start

### Export Comparison Table to LaTeX

```bash
# Basic export
python scripts/compare_ablation.py --latex docs/results_table.tex

# With custom caption and label
python scripts/compare_ablation.py \
  --latex docs/results_table.tex \
  --latex-caption "Comparison of Transfer Learning Strategies" \
  --latex-label "tab:transfer_learning"

# Or use the dedicated export script
python scripts/export_results_latex.py
```

## Usage Examples

### 1. Export Comparison Table

```bash
python scripts/compare_ablation.py \
  --outputs-dir outputs \
  --output ablation_comparison.csv \
  --latex docs/latex/comparison_table.tex \
  --latex-caption "Ablation Study: Transfer Learning Strategies" \
  --latex-label "tab:ablation_results"
```

This creates:
- CSV file: `ablation_comparison.csv`
- LaTeX file: `docs/latex/comparison_table.tex`

### 2. Include in Your LaTeX Document

In your milestone document (`milestone.tex` or similar):

```latex
\documentclass{article}
\usepackage{booktabs}  % For better table formatting

\begin{document}

% Include the generated table
\input{docs/latex/comparison_table.tex}

% Or reference it
Table~\ref{tab:ablation_results} shows the results...

\end{document}
```

### 3. Compile LaTeX Document

```bash
cd docs
pdflatex milestone.tex
pdflatex milestone.tex  # Run twice for references
```

## Generated LaTeX Format

The exported table will look like:

```latex
\begin{table}[htbp]
\centering
\begin{tabular}{lcccc}
\toprule
Experiment & Accuracy & Macro-F1 & Weighted-F1 & Underweight Recall \\
\midrule
scratch    & 0.3219   & 0.2987    & 0.3156       & 0.2845             \\
head_only  & 0.8543   & 0.8321    & 0.8498       & 0.7856             \\
last_block & 0.8721   & 0.8567    & 0.8698       & 0.8123             \\
full       & 0.8799   & 0.8650    & 0.8765       & 0.8234             \\
\bottomrule
\end{tabular}
\caption{Comparison of Transfer Learning Strategies}
\label{tab:ablation_results}
\end{table}
```

## Customization

### Modify Table Style

If you want to use `booktabs` package for nicer tables, edit `scripts/compare_ablation.py`:

```python
latex_str = df.to_latex(
    ...
    column_format="l" + "c" * (len(df.columns) - 1)
)
# Then add \toprule, \midrule, \bottomrule manually
```

### Include Per-Class Metrics

The script already includes per-class recall in the DataFrame. To export a separate table:

```python
# Create per-class table
per_class_df = pd.DataFrame({
    'Method': [...],
    'Recall 3.25': [...],
    'Recall 3.5': [...],
    ...
})
per_class_df.to_latex('docs/latex/per_class_recall.tex', ...)
```

## Troubleshooting

### Missing packages in LaTeX

If you get errors when compiling, you may need:

```latex
\usepackage{booktabs}  % For \toprule, \midrule, \bottomrule
\usepackage{graphicx}  % For \includegraphics (if including figures)
\usepackage{float}     % For [H] position specifier
```

Add to your LaTeX preamble.

### Unicode Characters

If you have issues with special characters, use:

```python
df.to_latex(escape=True)  # Escapes special LaTeX characters
```

### Table Too Wide

If table doesn't fit on page:

```latex
% Use adjustbox or resizebox
\usepackage{adjustbox}
\begin{adjustbox}{width=\textwidth}
\input{docs/latex/comparison_table.tex}
\end{adjustbox}
```

## Best Practices for Milestone

1. **Main Results Table**: Export with `--latex` flag
2. **Caption Clearly**: Use descriptive captions
3. **Consistent Formatting**: Use same decimal places (4) throughout
4. **Reference in Text**: Always reference tables: `Table~\ref{tab:ablation_results}`
5. **Include Units**: If needed, add units in column headers or caption

## Example Workflow

```bash
# 1. Run all experiments
bash scripts/run_ablation.sh

# 2. Generate comparison
python scripts/compare_ablation.py \
  --latex docs/latex/results.tex \
  --latex-caption "Transfer Learning Ablation Results" \
  --latex-label "tab:results"

# 3. Include in LaTeX document
# (manually add \input{docs/latex/results.tex} to your .tex file)

# 4. Compile
cd docs
pdflatex milestone.tex
```

## Additional Resources

- pandas `to_latex()` documentation: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_latex.html
- LaTeX table tutorial: https://www.overleaf.com/learn/latex/Tables

