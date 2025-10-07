import pandas as pd
from sklearn.model_selection import train_test_split
import pathlib

MANIFEST = pathlib.Path("data/manifest.csv")
df = pd.read_csv(MANIFEST)

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,          # FIXED seed for reproducibility
    stratify=df["bcs_5class"] # keep class proportions
)

train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)

print(f"Train: {len(train_df)}  Val: {len(val_df)}")
print(train_df["bcs_5class"].value_counts().sort_index())
print(val_df["bcs_5class"].value_counts().sort_index())
