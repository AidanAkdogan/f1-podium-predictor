# scripts/evaluate_features.py
import pandas as pd

df = pd.read_csv("artifacts/training_corpus.csv")

# === 1) NaN report ===
nan_report = df.isna().mean().sort_values(ascending=False)
print("\n📊 Features with >30% missing data:")
print(nan_report[nan_report > 0.3])

# === 2) Correlation with finish_position (numeric-only) ===

# Make sure finish_position is numeric
if "finish_position" in df.columns:
    df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
else:
    raise KeyError("'finish_position' column not found in training_corpus.csv")

# Keep only numeric columns for correlation
numeric_df = df.select_dtypes(include=["number"])

print("\n📊 Top 10 features correlated with finish_position:")
correlations = (
    numeric_df.corr()["finish_position"]
    .abs()
    .sort_values(ascending=False)
)
print(correlations.head(10))
