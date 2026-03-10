import pandas as pd
import numpy as np
from pathlib import Path

print("Starting FINAL robust headtracking processing...")

MAIN_FILE = "/home/ankita/BRSM/BRSM_Project1/360 Videos VR project/data/data.xlsx"
HT_BASE = Path("/home/ankita/BRSM/BRSM_Project1/360 Videos VR project/data/headtracking-data")
OUTPUT_MERGED = "merged_headtracking_final.xlsx"

df = pd.read_excel(MAIN_FILE)
df['dep_group'] = np.where(df['score_phq'] >= 10, 'Depressed', 'Non-depressed')

print(f"Main data loaded: {df.shape[0]} participants")
print("Sample v1 filenames:", df['v1'].head(3).tolist())

def clean_and_summarize(file_path):
    if not file_path.exists():
        print(f"  File missing: {file_path.name}")
        return None
    try:
        temp = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
        
        temp = temp[pd.to_numeric(temp.iloc[:, 0], errors='coerce').notna()]
        
        for col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors='coerce')
        
        temp = temp.dropna()
        
        if len(temp) < 10:
            print(f"  Too few rows in {file_path.name}")
            return None
            
        return {
            'mean_rot_speed': temp['RotationSpeedTotal'].mean(),
            'sd_yaw': temp['RotationChangeY'].std(ddof=1),
            'mean_yaw_speed': temp['RotationSpeedY'].mean(),
            'valid_rows': len(temp)
        }
    except Exception as e:
        print(f"Error in {file_path.name}: {e}")
        return None

records = []
for idx, row in df.iterrows():
    part_id = row['participant']
    part_summary = {'participant': part_id}
    
    speed_list = []
    yaw_list = []
    
    for v in range(1, 6):
        filename = row.get(f'v{v}')
        if pd.isna(filename):
            continue
        file_path = HT_BASE / f"v{v}" / str(filename)
        res = clean_and_summarize(file_path)
        if res:
            speed_list.append(res['mean_rot_speed'])
            yaw_list.append(res['sd_yaw'])
    
    if speed_list:
        part_summary['mean_rot_speed'] = np.mean(speed_list)
    if yaw_list:
        part_summary['sd_yaw'] = np.mean(yaw_list)
    
    records.append(part_summary)

ht_summary = pd.DataFrame(records)
merged = pd.merge(df, ht_summary, on='participant', how='left')

merged.to_excel(OUTPUT_MERGED, index=False)

print(f"\nSUCCESS! Final file saved: {OUTPUT_MERGED}")
print(f"Shape: {merged.shape} rows × {merged.shape[1]} columns")
print("\nFirst 8 rows of key columns:")
print(merged[['participant', 'score_phq', 'dep_group', 'mean_rot_speed', 'sd_yaw']].head(8))