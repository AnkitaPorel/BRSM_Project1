import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg

print("Loading data...")

df = pd.read_excel("/home/ankita/BRSM/BRSM_Project1/merged_headtracking_final.xlsx")

df['score_phq'] = pd.to_numeric(df['score_phq'], errors='coerce')
df['score_gad'] = pd.to_numeric(df['score_gad'], errors='coerce')

df['dep_group'] = np.where(df['score_phq'] >= 10, 'Depressed', 'Non-depressed')

df = df.dropna(subset=['mean_rot_speed', 'score_phq', 'score_gad']).copy()

print(f"Loaded {len(df)} participants | Depressed: {sum(df['dep_group']=='Depressed')}")

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='dep_group', y='mean_rot_speed', 
            hue='dep_group', palette='Set2', legend=False)
plt.title('Head Movement Speed by Depression Group')
plt.ylabel('Mean Rotation Speed')
plt.grid(axis='y', alpha=0.3)
plt.savefig('figure3_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Spearman Correlation (PHQ-9 vs Mean Rotation Speed) ===")
corr_result = pg.corr(df['score_phq'], df['mean_rot_speed'], method='spearman')
print(corr_result.round(4))

print("\n=== Welch t-test (Depressed vs Non-depressed) ===")
ttest = pg.ttest(df[df['dep_group']=='Depressed']['mean_rot_speed'],
                 df[df['dep_group']=='Non-depressed']['mean_rot_speed'], correction=True)
print(ttest.round(4))

print("\n=== ANCOVA controlling for GAD-7 ===")
ancova = pg.ancova(data=df, dv='mean_rot_speed', covar='score_gad', between='dep_group')
print(ancova.round(4))

print("\nAnalysis complete! Copy the printed numbers into your LaTeX report.")
print("Plot saved as: figure3_boxplot.png")
print("Tables ready in console — paste them directly into LaTeX.")