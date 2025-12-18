import pandas as pd

# Load the dataset
df = pd.read_excel('E:/KEM/Project/Data/raw/IMPUTED_DATA_WITH REDUCED_columns_21_09_2025.xlsx')

print(f"Dataset shape: {df.shape}")
print(f"\nTotal columns: {len(df.columns)}")

# Find columns that start with f1_ (potential data leakage)
f1_cols = [c for c in df.columns if c.startswith('f1_')]
print(f"\nColumns starting with 'f1_' (potential data leakage):")
for c in f1_cols:
    print(f"  {c}")

# Find columns containing LBW
lbw_cols = [c for c in df.columns if 'LBW' in c.upper()]
print(f"\nColumns containing 'LBW':")
for c in lbw_cols:
    print(f"  {c}")

# Target variable info
print(f"\nTarget variable (f1_bw) statistics:")
print(df['f1_bw'].describe())

# Missing values check
print(f"\nMissing values:")
missing = df.isnull().sum()
missing_cols = missing[missing > 0]
print(f"Columns with missing values: {len(missing_cols)}")
print(missing_cols.head(10))
