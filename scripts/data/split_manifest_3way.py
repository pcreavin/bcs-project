"""Create train/validation/test splits from manifest.csv."""
import pandas as pd
from sklearn.model_selection import train_test_split
import pathlib

MANIFEST = pathlib.Path("data/manifest.csv")
df = pd.read_csv(MANIFEST)

print(f"Total samples: {len(df)}")
print(f"Class distribution:\n{df['bcs_5class'].value_counts().sort_index()}")

# First split: 70% train, 30% temp (val + test)
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df["bcs_5class"]
)

# Second split: 30% → 15% val, 15% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,  # Split temp 50/50 to get 15% val, 15% test
    random_state=42,
    stratify=temp_df["bcs_5class"]
)

# Save splits
train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

# Print summary
print("\n" + "=" * 60)
print("SPLIT SUMMARY")
print("=" * 60)
print(f"Train: {len(train_df):,} samples ({100*len(train_df)/len(df):.1f}%)")
print(f"Val:   {len(val_df):,} samples ({100*len(val_df)/len(df):.1f}%)")
print(f"Test:  {len(test_df):,} samples ({100*len(test_df)/len(df):.1f}%)")
print(f"Total: {len(train_df) + len(val_df) + len(test_df):,} samples")

print("\nTrain class distribution:")
print(train_df["bcs_5class"].value_counts().sort_index())
print("\nVal class distribution:")
print(val_df["bcs_5class"].value_counts().sort_index())
print("\nTest class distribution:")
print(test_df["bcs_5class"].value_counts().sort_index())

print("\n" + "=" * 60)
print("Splits saved to:")
print("  - data/train.csv")
print("  - data/val.csv")
print("  - data/test.csv")
print("=" * 60)

