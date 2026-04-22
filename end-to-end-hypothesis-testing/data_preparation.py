"""
DATA PREPARATION:
- This file is used to prepare the data for the analysis including:
        - loading the data from the Excel file
        - creating a binary group label based on the PHQ-9 score
        - extracting the head-tracking features
        - building the per-participant averaged features for statistical analysis
        - saving the output to a CSV file
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EXCEL_FILE = os.path.join(DATA_DIR, "data.xlsx")
ALPHA      = 0.05

# All videos are considered for analysis
VIDEOS = ["v1", "v2", "v3", "v4", "v5"]

# All numeric features extracted per video (used for averaging across videos)
ALL_FEATURE_KEYS = [
    "mean_speed",
    "sd_pitch",   "sd_yaw",   "sd_roll",
    "mean_pitch", "mean_yaw", "mean_roll",
    "range_pitch","range_yaw","range_roll",
    "mean_speed_x","mean_speed_y","mean_speed_z",
    "total_movement",
    "n_samples",  "duration",
]

# Dependent variables for statistical testing (excludes recording metadata)
DV_COLS = [
    "mean_speed",
    "sd_pitch",   "sd_yaw",   "sd_roll",
    "mean_pitch", "mean_yaw", "mean_roll",
    "range_pitch","range_yaw","range_roll",
    "mean_speed_x","mean_speed_y","mean_speed_z",
    "total_movement",
]
BONF_ALPHA = ALPHA / len(DV_COLS)

DV_LABELS = {
    "mean_speed":   "Mean Speed (Total)",
    "sd_pitch":     "SD - Pitch (X)",
    "sd_yaw":       "SD - Yaw (Y)",
    "sd_roll":      "SD - Roll (Z)",
    "mean_pitch":   "Mean Rotation - Pitch (X)",
    "mean_yaw":     "Mean Rotation - Yaw (Y)",
    "mean_roll":    "Mean Rotation - Roll (Z)",
    "range_pitch":  "Range - Pitch (X)",
    "range_yaw":    "Range - Yaw (Y)",
    "range_roll":   "Range - Roll (Z)",
    "mean_speed_x": "Mean Speed - Pitch (X)",
    "mean_speed_y": "Mean Speed - Yaw (Y)",
    "mean_speed_z": "Mean Speed - Roll (Z)",
    "total_movement":"Total Movement Magnitude",
    "n_samples":     "N Samples",
    "duration":      "Duration",
}

# ── Helper: Extract features from full CSV time-series ────────────────────────
def extract_features(filepath):
    """
    Read the full head-tracking CSV and compute a comprehensive feature set
    from the raw time-series data.  Returns a dict or None on failure.
    """
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip', engine='python')
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna(how='all')

        required = ["RotationChangeX", "RotationChangeY", "RotationChangeZ",
                    "RotationSpeedX",  "RotationSpeedY",  "RotationSpeedZ",
                    "RotationSpeedTotal", "Time"]
        if not all(c in df.columns for c in required):
            return None

        df = df.dropna(subset=["RotationSpeedTotal"])

        return {
            "mean_speed":   df["RotationSpeedTotal"].mean(),
            "sd_pitch":     df["RotationChangeX"].std(),
            "sd_yaw":       df["RotationChangeY"].std(),
            "sd_roll":      df["RotationChangeZ"].std(),

            "mean_pitch":   df["RotationChangeX"].mean(),
            "mean_yaw":     df["RotationChangeY"].mean(),
            "mean_roll":    df["RotationChangeZ"].mean(),

            "range_pitch":  df["RotationChangeX"].max() - df["RotationChangeX"].min(),
            "range_yaw":    df["RotationChangeY"].max() - df["RotationChangeY"].min(),
            "range_roll":   df["RotationChangeZ"].max() - df["RotationChangeZ"].min(),

            "mean_speed_x": df["RotationSpeedX"].mean(),
            "mean_speed_y": df["RotationSpeedY"].mean(),
            "mean_speed_z": df["RotationSpeedZ"].mean(),

            "total_movement": np.sum(
                np.abs(df["RotationChangeX"]) +
                np.abs(df["RotationChangeY"]) +
                np.abs(df["RotationChangeZ"])
            ),

            "n_samples":    len(df),
            "duration":     df["Time"].max() - df["Time"].min(),
        }
    except Exception:
        return None


# ── Step 1: Load Excel and filter sample ─────────────────────────────────────
print("\n" + "="*70)
print("STEP 1 — LOADING DATA & FILTERING SAMPLE")
print("="*70)

df_main = pd.read_excel(EXCEL_FILE)
print(f"Full sample loaded: N = {len(df_main)}")

# Create binary group label
df_main["phq_group"]  = (df_main["score_phq"] >= 10).astype(int)
df_main["phq_label"]  = df_main["phq_group"].map({0: "Low (<10)", 1: "High (>=10)"})
print(f"  Group High PHQ (>=10): n = {df_main['phq_group'].sum()}")
print(f"  Group Low  PHQ (<10): n = {(df_main['phq_group']==0).sum()}")


# ── Step 2: Extract head-tracking features ───────────────────────────────────
print("\n" + "="*70)
print("STEP 2 — EXTRACTING HEAD-TRACKING FEATURES")
print("="*70)

missing_files = []

# 2a — Per-video feature columns appended to the original Excel dataframe
for vid in VIDEOS:
    vid_num = vid.replace("v", "")
    for key in ALL_FEATURE_KEYS:
        col_name = f"{DV_LABELS.get(key, key)}_v{vid_num}"
        df_main[col_name] = np.nan

for idx, row in df_main.iterrows():
    pid = row["participant"]
    for vid in VIDEOS:
        csv_filename = row[vid]
        if pd.isna(csv_filename):
            continue
        filepath = os.path.join(DATA_DIR, "headtracking-data", vid, csv_filename)
        feats = extract_features(filepath) if os.path.exists(filepath) else None

        if feats is None:
            missing_files.append(f"{pid} | {vid} | {csv_filename}")
            continue

        vid_num = vid.replace("v", "")
        for key in ALL_FEATURE_KEYS:
            col_name = f"{DV_LABELS.get(key, key)}_v{vid_num}"
            df_main.at[idx, col_name] = feats[key]

print(f"Per-video features appended to unified dataframe: "
      f"{len(VIDEOS)} videos × {len(ALL_FEATURE_KEYS)} features "
      f"= {len(VIDEOS) * len(ALL_FEATURE_KEYS)} new columns")
if missing_files:
    print(f"\n  ⚠ Missing files ({len(missing_files)}):")
    for mf in missing_files:
        print(f"    {mf}")

print("\n  Unified dataframe head():")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
print(df_main.head().to_string())
df_main.to_csv('output.csv', index=False)
pd.reset_option("display.max_columns")
pd.reset_option("display.width")

# 2b — Build per-participant averaged features for statistical analysis
records = []
for idx, row in df_main.iterrows():
    pid = row["participant"]
    vid_feats = []
    for vid in VIDEOS:
        vid_num = vid.replace("v", "")
        first_col = f"{DV_LABELS.get(ALL_FEATURE_KEYS[0], ALL_FEATURE_KEYS[0])}_v{vid_num}"
        if pd.notna(row[first_col]):
            vid_feats.append({
                key: row[f"{DV_LABELS.get(key, key)}_v{vid_num}"]
                for key in ALL_FEATURE_KEYS
            })

    if not vid_feats:
        continue

    avg = {key: np.mean([vf[key] for vf in vid_feats])
           for key in ALL_FEATURE_KEYS}
    avg["participant"]  = pid
    avg["score_phq"]    = row["score_phq"]
    avg["phq_group"]    = row["phq_group"]
    avg["phq_label"]    = row["phq_label"]
    avg["vr_experience"]= row["vr_experience"]
    avg["score_vrise"]  = row["score_vrise"]
    records.append(avg)

df_analysis = pd.DataFrame(records)
print(f"\nParticipants with extracted features: N = {len(df_analysis)}")

print("\nFeature summary (averaged across pleasant videos):")
print(df_analysis[DV_COLS].describe().round(3).to_string())