import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# --- 1. DATA PREP & FILTERING ---
df = pd.read_excel("data/data.xlsx")
df['TIME_start'] = pd.to_datetime(df['TIME_start'])
df['TIME_end'] = pd.to_datetime(df['TIME_end'])

# Filter VRISE < 25
df = df[df['score_vrise'] >= 25].copy()
print(f"Working with N={len(df)} participants.\n")

# --- 2. EXTRACT HEADTRACKING METRICS PER VIDEO ---
videos = ['v1', 'v2', 'v3', 'v4', 'v5']
base_dir = "data/headtracking-data"

# Initialize dictionaries to store our aggregated data
ht_data = []

print("Extracting headtracking metrics for EDA...")

for index, row in df.iterrows():
    start_time = row['TIME_start']
    end_time = row['TIME_end']
    participant_id = row['participant']
    
    for v in videos:
        folder_path = os.path.join(base_dir, v)
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        for file in csv_files:
            filename = os.path.basename(file)
            time_str = filename.split('_')[2].split('.')[0][:14]
            file_time = datetime.strptime(time_str, "%Y%m%d%H%M%S")
            
            if start_time <= file_time <= end_time:
                try:
                    df_ht = pd.read_csv(file, on_bad_lines='skip')
                    
                    # Calculate descriptive metrics for this specific video session
                    mean_speed = df_ht['RotationSpeedTotal'].mean()
                    sd_pitch = df_ht['RotationChangeX'].std() # Up/Down spread
                    sd_yaw = df_ht['RotationChangeY'].std()   # Left/Right spread
                    sd_roll = df_ht['RotationChangeZ'].std()  # Tilt spread
                    
                    ht_data.append({
                        'Participant': participant_id,
                        'Video': v.upper(),
                        'Mean_Speed': mean_speed,
                        'SD_Pitch_X': sd_pitch,
                        'SD_Yaw_Y': sd_yaw,
                        'SD_Roll_Z': sd_roll
                    })
                except Exception as e:
                    pass
                break # Found the file, move to next video

# Convert to DataFrame for easy analysis
df_metrics = pd.DataFrame(ht_data)

# --- 3. DESCRIPTIVE STATISTICS PER VIDEO ---
print("\n=== HEADTRACKING DESCRIPTIVE STATS BY VIDEO ===")
# Group by video and calculate mean of our metrics
stats_by_video = df_metrics.groupby('Video')[['Mean_Speed', 'SD_Pitch_X', 'SD_Yaw_Y']].mean().round(3)
print(stats_by_video)

# --- 4. VISUALIZATIONS ---
sns.set_theme(style="whitegrid", palette="Set2")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Objective Headtracking Responses Across 360° Videos', fontsize=16)

# Plot 1: Mean Rotation Speed per Video
sns.boxplot(data=df_metrics, x='Video', y='Mean_Speed', ax=axes[0], order=['V1', 'V2', 'V3', 'V4', 'V5'])
sns.stripplot(data=df_metrics, x='Video', y='Mean_Speed', ax=axes[0], color=".25", alpha=0.5, order=['V1', 'V2', 'V3', 'V4', 'V5'])
axes[0].set_title('Average Head Movement Speed')
axes[0].set_ylabel('Mean RotationSpeedTotal')

# Plot 2: Horizontal Scanning (Yaw) per Video
sns.boxplot(data=df_metrics, x='Video', y='SD_Yaw_Y', ax=axes[1], order=['V1', 'V2', 'V3', 'V4', 'V5'])
sns.stripplot(data=df_metrics, x='Video', y='SD_Yaw_Y', ax=axes[1], color=".25", alpha=0.5, order=['V1', 'V2', 'V3', 'V4', 'V5'])
axes[1].set_title('Horizontal Scanning Range (SD of Yaw)')
axes[1].set_ylabel('Standard Deviation of Rotation Y')

plt.tight_layout()
plt.savefig("eda_plot4_headtracking_by_video.png")

print("\nVisualization saved as 'eda_plot4_headtracking_by_video.png'")
plt.show()
