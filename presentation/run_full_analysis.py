import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pingouin as pg
import warnings
warnings.filterwarnings('ignore')

# --- 1. LOAD MAIN DATA & FILTER OUTLIERS ---
df_main = pd.read_excel("data/data.xlsx")
# Ensure datetime format
df_main['TIME_start'] = pd.to_datetime(df_main['TIME_start'])
df_main['TIME_end'] = pd.to_datetime(df_main['TIME_end'])

# Exclude participants with severe simulator sickness (VRISE < 25)
initial_n = len(df_main)
df_main = df_main[df_main['score_vrise'] >= 25].copy()
print(f"Excluded {initial_n - len(df_main)} participants due to VRISE < 25. Remaining N={len(df_main)}")

# Create PHQ Groups (None vs Mild+)
df_main['PHQ_Group'] = np.where(df_main['score_phq'] >= 5, 'Depressed (Mild+)', 'Healthy Control')

# --- 2. MAP CSVs AND EXTRACT FEATURES ---
videos = ['v1', 'v2', 'v3', 'v4', 'v5']
base_dir = "data/headtracking-data"

# Initialize columns to store our new metrics
for v in videos:
    df_main[f'{v}_mean_speed'] = np.nan
    df_main[f'{v}_sd_yaw'] = np.nan # Yaw = Rotation Y (horizontal scanning)

print("\nExtracting headtracking features (this may take a moment)...")

for index, row in df_main.iterrows():
    start_time = row['TIME_start']
    end_time = row['TIME_end']
    
    for v in videos:
        folder_path = os.path.join(base_dir, v)
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        for file in csv_files:
            # Extract timestamp from filename: data_video1_YYYYMMDDHHMMSSmmm.csv
            filename = os.path.basename(file)
            time_str = filename.split('_')[2].split('.')[0][:14] # Get YYYYMMDDHHMMSS
            file_time = datetime.strptime(time_str, "%Y%m%d%H%M%S")
            
            # If the CSV timestamp falls within the participant's session
            if start_time <= file_time <= end_time:
                try:
                    df_ht = pd.read_csv(file, on_bad_lines='skip')
                    # Calculate Metrics
                    mean_speed = df_ht['RotationSpeedTotal'].mean()
                    sd_yaw = df_ht['RotationChangeY'].std() # Range of horizontal looking
                    
                    # Store in main dataframe
                    df_main.loc[index, f'{v}_mean_speed'] = mean_speed
                    df_main.loc[index, f'{v}_sd_yaw'] = sd_yaw
                except Exception as e:
                    pass
                break # Move to next video once found

# Calculate an overall average psychomotor speed across all 5 videos
speed_cols = [f'{v}_mean_speed' for v in videos]
df_main['overall_mean_speed'] = df_main[speed_cols].mean(axis=1)

# --- 3. DESCRIPTIVE & INFERENTIAL STATISTICS ---

print("\n=== T-TEST: PSYCHOMOTOR SPEED BY GROUP ===")
hc_speed = df_main[df_main['PHQ_Group'] == 'Healthy Control']['overall_mean_speed'].dropna()
dep_speed = df_main[df_main['PHQ_Group'] == 'Depressed (Mild+)']['overall_mean_speed'].dropna()

t_stat, p_val = stats.ttest_ind(hc_speed, dep_speed)
print(f"Healthy (M={hc_speed.mean():.2f}) vs Depressed (M={dep_speed.mean():.2f})")
print(f"T-statistic: {t_stat:.3f}, p-value: {p_val:.3f}")

print("\n=== PARTIAL CORRELATION (Isolating Depression) ===")
# Correlation between Depression and Speed, controlling for Anxiety (GAD-7)
pcorr = pg.partial_corr(data=df_main, x='score_phq', y='overall_mean_speed', covar='score_gad')
print("Partial Correlation (PHQ-9 & Speed, controlling for GAD-7):")
# Dropping the specific column selection to avoid KeyErrors
print(pcorr.round(3))

# --- 4. VISUALIZATIONS ---
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Boxplot of Groups
sns.boxplot(data=df_main, x='PHQ_Group', y='overall_mean_speed', ax=axes[0], palette="Set2")
sns.swarmplot(data=df_main, x='PHQ_Group', y='overall_mean_speed', ax=axes[0], color=".25")
axes[0].set_title('Average Head Movement Speed by Depression Group')
axes[0].set_ylabel('Mean Rotation Speed')
axes[0].set_xlabel('')

# Plot 2: Scatter plot of Continuous PHQ-9 vs Speed
sns.regplot(data=df_main, x='score_phq', y='overall_mean_speed', ax=axes[1], scatter_kws={'alpha':0.6}, color='b')
axes[1].set_title('Continuous Relationship: Depressive Symptoms vs Speed')
axes[1].set_ylabel('Mean Rotation Speed')
axes[1].set_xlabel('PHQ-9 Score')

plt.tight_layout()
plt.savefig("psychomotor_analysis_results.png")
print("\nVisualization saved as 'psychomotor_analysis_results.png'")
plt.show()
