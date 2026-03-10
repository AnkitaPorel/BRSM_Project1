import pandas as pd

# 1. Safely Inspect the Headtracking CSV
sample_csv_path = "data/headtracking-data/v1/data_video1_20260125113153995.csv"
print("=== HEADTRACKING CSV STRUCTURE ===")
try:
    # on_bad_lines='skip' will ignore rows with structural errors
    df_csv = pd.read_csv(sample_csv_path, on_bad_lines='skip')
    print(f"Shape (excluding bad lines): {df_csv.shape}")
    print(f"Columns: {list(df_csv.columns)}")
    print("\nFirst 2 rows:")
    print(df_csv.head(2))
except Exception as e:
    print(f"Error reading CSV: {e}")

print("\n" + "="*50 + "\n")

# 2. Load Main Data for Preliminary Descriptive Stats
excel_path = "data/data.xlsx"
df_main = pd.read_excel(excel_path)

print("=== DESCRIPTIVE STATISTICS (N=40) ===")
stats_cols = ['age', 'score_phq', 'score_gad', 'score_stai_t', 'score_vrise']
# Display count, mean, std, min, max, and quartiles
print(df_main[stats_cols].describe().round(2))

print("\n" + "="*50 + "\n")

# 3. Check Covariance/Correlation of Depression and Anxiety
print("=== PEARSON CORRELATION (Mental Health Scores) ===")
corr_matrix = df_main[['score_phq', 'score_gad', 'score_stai_t']].corr().round(3)
print(corr_matrix)

# 4. Check PHQ-9 Distribution to Help Decide Group Partitioning
print("\n=== PHQ-9 (Depression) SCORE DISTRIBUTION ===")
# Clinical Cutoffs for PHQ-9: 0-4 (None), 5-9 (Mild), 10-14 (Moderate), 15-19 (Moderately Severe), 20-27 (Severe)
bins = [-1, 4, 9, 14, 19, 27]
labels = ['Minimal (0-4)', 'Mild (5-9)', 'Moderate (10-14)', 'Mod-Severe (15-19)', 'Severe (20-27)']
df_main['phq_category'] = pd.cut(df_main['score_phq'], bins=bins, labels=labels)
print(df_main['phq_category'].value_counts().sort_index())
