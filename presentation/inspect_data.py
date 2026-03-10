import pandas as pd
import os

# 1. Print all column names in the main dataset
excel_path = "data/data.xlsx"
print("=== data.xlsx COLUMNS ===")
try:
    df_main = pd.read_excel(excel_path)
    print(list(df_main.columns))
except Exception as e:
    print(f"Error reading {excel_path}: {e}")

print("\n" + "="*50 + "\n")

# 2. Inspect a sample headtracking CSV file (Video 1)
sample_csv_path = "data/headtracking-data/v1/data_video1_20260125113153995.csv"
print(f"=== SAMPLE CSV STRUCTURE ({os.path.basename(sample_csv_path)}) ===")
try:
    df_csv = pd.read_csv(sample_csv_path)
    print(f"Shape: {df_csv.shape}")
    print("\nColumns and Data Types:")
    print(df_csv.dtypes)
    print("\nFirst 3 rows:")
    print(df_csv.head(3))
except Exception as e:
    print(f"Error reading {sample_csv_path}: {e}")
