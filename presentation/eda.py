import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# --- 1. DATA PREP & CLEANING ---
df = pd.read_excel("data/data.xlsx")

# Apply our exclusion criteria (VRISE < 25)
initial_n = len(df)
df = df[df['score_vrise'] >= 25].copy()
print(f"=== DATA CLEANING ===")
print(f"Initial N: {initial_n} | Excluded (VRISE < 25): {initial_n - len(df)} | Final N: {len(df)}\n")

# --- 2. DEMOGRAPHICS & DESCRIPTIVE STATS ---
print("=== DEMOGRAPHICS ===")
print(df[['age', 'gender', 'vr_experience']].describe().round(2))
print("\nGender Counts (1=Male, 2=Female, etc. - adjust mapping as needed):")
print(df['gender'].value_counts())

print("\n=== MENTAL HEALTH INVENTORIES (Descriptive Stats) ===")
mh_cols = ['score_phq', 'score_gad', 'score_stai_t', 'positive_affect_start', 'negative_affect_start']
print(df[mh_cols].describe().round(2))

# --- 3. INFERENTIAL PRELIMINARY STATS (CORRELATIONS) ---
print("\n=== CORRELATION MATRIX (Pearson) ===")
corr_matrix = df[mh_cols].corr().round(3)
print(corr_matrix)

# --- 4. VISUALIZATIONS ---
sns.set_theme(style="whitegrid", palette="muted")

# Plot 1: Distributions of Mental Health Variables
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
fig1.suptitle('Distributions of Key Mental Health Inventories', fontsize=16)

sns.histplot(df['score_phq'], kde=True, ax=axes1[0], color='blue', bins=10)
axes1[0].set_title('PHQ-9 (Depression)')

sns.histplot(df['score_gad'], kde=True, ax=axes1[1], color='orange', bins=10)
axes1[1].set_title('GAD-7 (Anxiety)')

sns.histplot(df['score_stai_t'], kde=True, ax=axes1[2], color='green', bins=10)
axes1[2].set_title('STAI-T (Trait Anxiety)')

plt.tight_layout()
plt.savefig("eda_plot1_distributions.png")

# Plot 2: Correlation Heatmap
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
ax2.set_title('Correlation Heatmap of Surveys', fontsize=14)
plt.tight_layout()
plt.savefig("eda_plot2_correlations.png")

# Plot 3: Video Comparisons (Subjective Ratings)
# We need to melt the dataframe to easily plot V1-V5 side by side
valence_cols = ['valence_v1', 'valence_v2', 'valence_v3', 'valence_v4', 'valence_v5']
arousal_cols = ['arousal_v1', 'arousal_v2', 'arousal_v3', 'arousal_v4', 'arousal_v5']

df_valence = df[valence_cols].melt(var_name='Video', value_name='Valence')
df_arousal = df[arousal_cols].melt(var_name='Video', value_name='Arousal')

# Clean up video labels for the plot
df_valence['Video'] = df_valence['Video'].str.replace('valence_', '').str.upper()
df_arousal['Video'] = df_arousal['Video'].str.replace('arousal_', '').str.upper()

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle('Subjective Ratings Across 360-degree Videos', fontsize=16)

sns.boxplot(data=df_valence, x='Video', y='Valence', ax=axes3[0], palette="Blues")
sns.swarmplot(data=df_valence, x='Video', y='Valence', ax=axes3[0], color=".25", alpha=0.6)
axes3[0].set_title('Valence (Pleasantness)')
axes3[0].set_ylim(0, 10) # Assuming 9-point scale, padding for visual

sns.boxplot(data=df_arousal, x='Video', y='Arousal', ax=axes3[1], palette="Reds")
sns.swarmplot(data=df_arousal, x='Video', y='Arousal', ax=axes3[1], color=".25", alpha=0.6)
axes3[1].set_title('Arousal (Excitement/Intensity)')
axes3[1].set_ylim(0, 10)

plt.tight_layout()
plt.savefig("eda_plot3_video_subjective.png")

print("\nEDA plots saved successfully: ")
print("1. eda_plot1_distributions.png")
print("2. eda_plot2_correlations.png")
print("3. eda_plot3_video_subjective.png")

plt.show()
