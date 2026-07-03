from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_table.csv"

df = pd.read_csv(FEATURE_PATH)

print(df.info())

print("\nFirst five rows:\n")
print(df.head())

print("\nMissing values:\n")
print(df.isnull().sum().sort_values(ascending=False).head(20))
