#!/usr/bin/env python3
"""
VR Immersion Prediction from Head-Tracking Data
Fixed: CSV parsing, dummy variable conversion, and NaN handling.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
import statsmodels.formula.api as smf
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = "/home/ankita/BRSM/BRSM_Project1/360 Videos VR project/data"
EXCEL_PATH = os.path.join(BASE_DIR, "data.xlsx")
HT_DIR = os.path.join(BASE_DIR, "headtracking-data")

VIDEO_LABELS = {1: "alley", 2: "beach", 3: "campus", 4: "horror", 5: "surfing"}

# =============================================================================
# 1. Read survey data from Excel
# =============================================================================
print("Reading survey data from Excel...")
survey = pd.read_excel(EXCEL_PATH, sheet_name="Sheet1")
survey.rename(columns={survey.columns[0]: "participant"}, inplace=True)

# =============================================================================
# 2. Reshape to long format
# =============================================================================
print("Reshaping to long format...")
records = []
for _, row in survey.iterrows():
    for vid in range(1, 6):
        rec = {
            "participant": row["participant"],
            "video": vid,
            "valence": row[f"valence_v{vid}"],
            "arousal": row[f"arousal_v{vid}"],
            "immersion": row[f"immersion_v{vid}"],
            "csv_file": row[f"v{vid}"],
            "age": row["age"],
            "gender": row["gender"],
            "vr_experience": row["vr_experience"],
            "score_stai_t": row["score_stai_t"],
            "score_phq": row["score_phq"],
            "score_gad": row["score_gad"],
            "score_vrise": row["score_vrise"],
            "positive_affect_start": row["positive_affect_start"],
            "negative_affect_start": row["negative_affect_start"],
        }
        records.append(rec)

survey_long = pd.DataFrame(records)
survey_long["video_type"] = survey_long["video"].map(VIDEO_LABELS)

# =============================================================================
# 3. Function to extract motion features
# =============================================================================
def extract_motion_features(csv_path):
    """Return a dict of features, or None if file missing/invalid."""
    if not os.path.exists(csv_path):
        print(f"  Warning: File not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip', engine='c')
    except Exception:
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip', engine='python')
        except Exception as e:
            print(f"  Error reading {csv_path}: {e}")
            return None
    
    # Convert Time column to numeric and drop rows where conversion fails
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df = df.dropna(subset=['Time'])
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['Time'])
    
    if len(df) < 2:
        return None
    
    features = {}
    features["mean_rot_speed_total"] = df["RotationSpeedTotal"].mean()
    features["sd_rot_speed_total"]   = df["RotationSpeedTotal"].std()
    features["max_rot_speed_total"]  = df["RotationSpeedTotal"].max()
    features["p95_rot_speed_total"]  = df["RotationSpeedTotal"].quantile(0.95)
    features["total_angular_y"]      = df["RotationChangeY"].abs().sum()
    features["total_position"]       = np.sqrt(df["PositionChangeX"]**2 + 
                                               df["PositionChangeY"]**2 + 
                                               df["PositionChangeZ"]**2).sum()
    
    sign_changes = np.diff(np.sign(df["RotationChangeY"])) != 0
    features["dir_changes_y"] = sign_changes.sum()
    
    if len(df) > 2:
        X_slope = sm.add_constant(df["Time"])
        model_slope = sm.OLS(df["RotationSpeedTotal"], X_slope).fit()
        features["rot_speed_slope"] = model_slope.params.iloc[1]
    else:
        features["rot_speed_slope"] = np.nan
    
    return features

# =============================================================================
# 4. Process all CSV files
# =============================================================================
print("Extracting motion features from CSVs...")
all_features = []

for idx, row in survey_long.iterrows():
    participant = row["participant"]
    video = row["video"]
    csv_name = row["csv_file"]
    
    csv_path = os.path.join(HT_DIR, f"v{video}", csv_name)
    feat = extract_motion_features(csv_path)
    
    if feat is not None:
        feat["participant"] = participant
        feat["video"] = video
        all_features.append(feat)

features_df = pd.DataFrame(all_features)
print(f"  Extracted features for {len(features_df)} video sessions.")

# =============================================================================
# 5. Merge features with survey data
# =============================================================================
if features_df.empty:
    print("No features extracted. Exiting.")
    exit(1)

model_data = survey_long.merge(features_df, on=["participant", "video"], how="inner")
print(f"Merged data: {len(model_data)} rows.")

# Scale continuous predictors
def scale_series(series):
    return (series - series.mean()) / series.std()

model_data["mean_rot_speed_total_s"] = scale_series(model_data["mean_rot_speed_total"])
model_data["sd_rot_speed_total_s"]   = scale_series(model_data["sd_rot_speed_total"])
model_data["total_angular_y_s"]      = scale_series(model_data["total_angular_y"])
model_data["age_s"]                  = scale_series(model_data["age"])
model_data["score_stai_t_s"]         = scale_series(model_data["score_stai_t"])

# =============================================================================
# 5.5. Correlation Matrix of Continuous Predictors
# =============================================================================
print("\nGenerating correlation matrix...")
continuous_predictors = [
    "mean_rot_speed_total_s", "sd_rot_speed_total_s", 
    "total_angular_y_s", "age_s", "score_stai_t_s", "vr_experience"
]

corr_matrix = model_data[continuous_predictors].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix of Continuous Predictors", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=150)
plt.show()
print("Correlation matrix saved as 'correlation_matrix.png'")

# =============================================================================
# 6. Build Regression Model (OLS with participant fixed effects)
# =============================================================================
print("\nFitting regression model...")

# Create dummy variables and convert to int
participant_dummies = pd.get_dummies(model_data["participant"], prefix="p", drop_first=False).astype(int)
video_dummies = pd.get_dummies(model_data["video_type"], prefix="vid", drop_first=False).astype(int)

# Combine predictors
X = pd.concat([
    model_data[["mean_rot_speed_total_s", "sd_rot_speed_total_s", 
                "total_angular_y_s", "age_s", "score_stai_t_s", "vr_experience"]],
    video_dummies,
    participant_dummies
], axis=1)

# Clean column names (replace '.' with '_' to avoid statsmodels formula issues)
X.columns = [col.replace('.', '_') for col in X.columns]

# Add intercept
X = sm.add_constant(X)

y = model_data["immersion"]

# Drop rows with any NaN
valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
X = X[valid_idx]
y = y[valid_idx]

# Convert to float
X = X.astype(float)
y = y.astype(float)

print(f"Data shape after cleaning: {X.shape}")

# Fit OLS
model = sm.OLS(y, X).fit()

# =============================================================================
# 6.5. Multicollinearity Check (VIF)
# =============================================================================
print("\n" + "="*60)
print("VARIANCE INFLATION FACTOR (VIF) - Multicollinearity Check")
print("="*60)
print("Rule of thumb: VIF > 10 indicates high multicollinearity")
print("              VIF > 5 may warrant investigation\n")

# Calculate VIF for continuous predictors (exclude intercept and dummies)
vif_data = pd.DataFrame()
vif_predictors = [col for col in X.columns if col in continuous_predictors]

if len(vif_predictors) > 0:
    X_vif = X[vif_predictors]
    vif_data["Predictor"] = vif_predictors
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) 
                       for i in range(len(vif_predictors))]
    vif_data = vif_data.sort_values("VIF", ascending=False)
    print(vif_data.to_string(index=False))
else:
    print("No continuous predictors found for VIF calculation")

# =============================================================================
# 6.6. Null Model Comparison
# =============================================================================
print("\n" + "="*60)
print("NULL MODEL COMPARISON")
print("="*60)

# Fit null model (intercept only)
X_null = sm.add_constant(pd.DataFrame(np.ones(len(y))))
model_null = sm.OLS(y, X_null).fit()

print("\n--- NULL MODEL (Intercept Only) ---")
print(f"R-squared:          {model_null.rsquared:.6f}")
print(f"Adj. R-squared:     {model_null.rsquared_adj:.6f}")
print(f"AIC:                {model_null.aic:.2f}")
print(f"BIC:                {model_null.bic:.2f}")
print(f"Log-Likelihood:     {model_null.llf:.2f}")

print("\n--- FULL MODEL ---")
print(f"R-squared:          {model.rsquared:.6f}")
print(f"Adj. R-squared:     {model.rsquared_adj:.6f}")
print(f"AIC:                {model.aic:.2f}")
print(f"BIC:                {model.bic:.2f}")
print(f"Log-Likelihood:     {model.llf:.2f}")

print("\n--- MODEL COMPARISON ---")
r2_improvement = model.rsquared - model_null.rsquared
adj_r2_improvement = model.rsquared_adj - model_null.rsquared_adj
print(f"R² improvement:     {r2_improvement:.6f} ({r2_improvement*100:.2f}%)")
print(f"Adj. R² improvement: {adj_r2_improvement:.6f} ({adj_r2_improvement*100:.2f}%)")

# Likelihood ratio test
lr_statistic = 2 * (model.llf - model_null.llf)
df_diff = model.df_model - model_null.df_model
lr_pvalue = stats.chi2.sf(lr_statistic, df_diff)
print(f"\nLikelihood Ratio Test:")
print(f"  Chi-square statistic: {lr_statistic:.4f}")
print(f"  Degrees of freedom:   {df_diff}")
print(f"  P-value:              {lr_pvalue:.6e}")
if lr_pvalue < 0.001:
    print(f"  Result: Full model is significantly better than null (p < 0.001) ***")
elif lr_pvalue < 0.01:
    print(f"  Result: Full model is significantly better than null (p < 0.01) **")
elif lr_pvalue < 0.05:
    print(f"  Result: Full model is significantly better than null (p < 0.05) *")
else:
    print(f"  Result: No significant improvement over null model")

# =============================================================================
# 6.7. Forward Stepwise Regression
# =============================================================================
print("\n" + "="*60)
print("FORWARD STEPWISE REGRESSION")
print("="*60)
print("Building model by adding one predictor at a time...")
print("Selection criterion: Lowest AIC at each step\n")

# Get list of candidate predictors (exclude intercept and fixed effects for simplicity)
candidate_predictors = continuous_predictors.copy()

# Track the stepwise process
stepwise_results = []
selected_predictors = []
remaining_predictors = candidate_predictors.copy()

# Starting model: intercept only
current_predictors = []
best_aic = model_null.aic

step = 0
print(f"Step {step}: NULL MODEL (Intercept only)")
print(f"  AIC: {model_null.aic:.2f}")
print(f"  R²: {model_null.rsquared:.6f}")
print(f"  Adj. R²: {model_null.rsquared_adj:.6f}\n")

stepwise_results.append({
    'Step': step,
    'Predictor_Added': 'Intercept',
    'AIC': model_null.aic,
    'BIC': model_null.bic,
    'R_squared': model_null.rsquared,
    'Adj_R_squared': model_null.rsquared_adj,
    'Num_Predictors': 0
})

# Forward selection loop
while remaining_predictors:
    step += 1
    best_candidate = None
    best_candidate_aic = np.inf
    best_candidate_model = None
    
    # Try adding each remaining predictor
    for candidate in remaining_predictors:
        test_predictors = selected_predictors + [candidate]
        
        # Build model with current selected predictors + candidate
        X_step = sm.add_constant(model_data[test_predictors])
        
        # Handle NaN values
        valid_idx_step = ~(X_step.isnull().any(axis=1) | y.isnull())
        X_step_clean = X_step[valid_idx_step].astype(float)
        y_step_clean = y[valid_idx_step].astype(float)
        
        # Fit model
        try:
            model_step = sm.OLS(y_step_clean, X_step_clean).fit()
            
            # Check if this is the best candidate so far
            if model_step.aic < best_candidate_aic:
                best_candidate = candidate
                best_candidate_aic = model_step.aic
                best_candidate_model = model_step
        except:
            continue
    
    # Check if adding the best candidate improves AIC
    if best_candidate is not None and best_candidate_aic < best_aic:
        selected_predictors.append(best_candidate)
        remaining_predictors.remove(best_candidate)
        best_aic = best_candidate_aic
        
        print(f"Step {step}: Added '{best_candidate}'")
        print(f"  AIC: {best_candidate_model.aic:.2f} (Δ = {best_candidate_model.aic - stepwise_results[-1]['AIC']:.2f})")
        print(f"  R²: {best_candidate_model.rsquared:.6f} (Δ = {best_candidate_model.rsquared - stepwise_results[-1]['R_squared']:.6f})")
        print(f"  Adj. R²: {best_candidate_model.rsquared_adj:.6f}")
        
        # Get p-value for the newly added predictor
        if best_candidate in best_candidate_model.pvalues.index:
            p_val = best_candidate_model.pvalues[best_candidate]
            sig_stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            print(f"  p-value: {p_val:.6f} {sig_stars}\n")
        else:
            print()
        
        stepwise_results.append({
            'Step': step,
            'Predictor_Added': best_candidate,
            'AIC': best_candidate_model.aic,
            'BIC': best_candidate_model.bic,
            'R_squared': best_candidate_model.rsquared,
            'Adj_R_squared': best_candidate_model.rsquared_adj,
            'Num_Predictors': len(selected_predictors)
        })
    else:
        print(f"Step {step}: No improvement found. Stopping.\n")
        break

# Create summary table
stepwise_df = pd.DataFrame(stepwise_results)
stepwise_df['Delta_R2'] = stepwise_df['R_squared'].diff().fillna(0)
stepwise_df['Delta_AIC'] = stepwise_df['AIC'].diff().fillna(0)

print("\n--- STEPWISE REGRESSION SUMMARY ---")
print(stepwise_df.to_string(index=False))

# Save stepwise results
stepwise_df.to_csv("stepwise_regression_results.csv", index=False)
print("\nStepwise results saved as 'stepwise_regression_results.csv'")

# Plot R² progression
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(stepwise_df['Step'], stepwise_df['R_squared'], marker='o', linewidth=2, markersize=8)
plt.plot(stepwise_df['Step'], stepwise_df['Adj_R_squared'], marker='s', linewidth=2, markersize=8)
plt.xlabel('Step', fontsize=12)
plt.ylabel('R²', fontsize=12)
plt.title('R² Progression in Forward Stepwise Regression', fontsize=14, fontweight='bold')
plt.legend(['R²', 'Adjusted R²'], fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(stepwise_df['Step'], stepwise_df['AIC'], marker='o', linewidth=2, markersize=8, color='red')
plt.xlabel('Step', fontsize=12)
plt.ylabel('AIC', fontsize=12)
plt.title('AIC Progression in Forward Stepwise Regression', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("stepwise_progression.png", dpi=150)
plt.show()
print("Stepwise progression plot saved as 'stepwise_progression.png'")

print(f"\n--- FINAL STEPWISE MODEL ---")
print(f"Selected predictors ({len(selected_predictors)}): {selected_predictors}")
print(f"Final R²: {stepwise_df.iloc[-1]['R_squared']:.6f}")
print(f"Final Adj. R²: {stepwise_df.iloc[-1]['Adj_R_squared']:.6f}")
print(f"Final AIC: {stepwise_df.iloc[-1]['AIC']:.2f}")

# =============================================================================
# 8. Output
# =============================================================================
print("\n" + "="*60)
print("MODEL SUMMARY")
print("="*60)
print(model.summary())

# Save model and data
model.save("immersion_ols_model.pickle")
model_data.to_csv("model_data.csv", index=False)
print("\nModel saved as 'immersion_ols_model.pickle'")
print("Merged data saved as 'model_data.csv'")

# =============================================================================
# 9. Diagnostic Plots
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0,0].scatter(model.fittedvalues, model.resid, alpha=0.5)
axes[0,0].axhline(0, color='red', linestyle='--')
axes[0,0].set_xlabel("Fitted Values")
axes[0,0].set_ylabel("Residuals")
axes[0,0].set_title("Residuals vs Fitted")

sm.qqplot(model.resid, line='s', ax=axes[0,1])
axes[0,1].set_title("Normal Q-Q")

axes[1,0].scatter(model.fittedvalues, np.sqrt(np.abs(model.resid)), alpha=0.5)
axes[1,0].set_xlabel("Fitted Values")
axes[1,0].set_ylabel("√|Residuals|")
axes[1,0].set_title("Scale-Location")

axes[1,1].scatter(model.fittedvalues, y, alpha=0.5)
axes[1,1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
axes[1,1].set_xlabel("Predicted Immersion")
axes[1,1].set_ylabel("Observed Immersion")
axes[1,1].set_title("Predicted vs Observed")

plt.tight_layout()
plt.savefig("model_diagnostics.png", dpi=150)
plt.show()

print("\nDiagnostic plots saved as 'model_diagnostics.png'")

# =============================================================================
# 9.5. Cook's Distance - Influential Outliers Detection
# =============================================================================
print("\n" + "="*60)
print("COOK'S DISTANCE - Influential Outliers Detection")
print("="*60)

# Calculate Cook's distance
influence = model.get_influence()
cooks_d = influence.cooks_distance[0]

# Common threshold: 4/n or 4/(n-k-1)
n = len(y)
k = model.df_model
threshold_1 = 4 / n
threshold_2 = 4 / (n - k - 1)

print(f"Sample size (n): {n}")
print(f"Number of predictors (k): {k}")
print(f"Cook's D threshold (4/n): {threshold_1:.6f}")
print(f"Cook's D threshold (4/(n-k-1)): {threshold_2:.6f}")

# Identify influential points
influential_threshold = threshold_2  # Use the more conservative threshold
influential_points = np.where(cooks_d > influential_threshold)[0]

print(f"\nInfluential observations (Cook's D > {influential_threshold:.6f}): {len(influential_points)}")

if len(influential_points) > 0:
    # Create dataframe of influential points
    influential_df = pd.DataFrame({
        'Index': influential_points,
        'Cooks_D': cooks_d[influential_points],
        'Participant': model_data.iloc[influential_points]['participant'].values,
        'Video': model_data.iloc[influential_points]['video'].values,
        'Immersion': y.iloc[influential_points].values,
        'Fitted': model.fittedvalues.iloc[influential_points].values,
        'Residual': model.resid.iloc[influential_points].values
    })
    influential_df = influential_df.sort_values('Cooks_D', ascending=False)
    
    print(f"\nTop {min(10, len(influential_points))} most influential observations:")
    print(influential_df.head(10).to_string(index=False))
    
    # Save full list
    influential_df.to_csv("influential_observations.csv", index=False)
    print(f"\nFull list saved as 'influential_observations.csv'")
else:
    print("\nNo influential observations detected.")

# Plot Cook's distance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.stem(range(len(cooks_d)), cooks_d, markerfmt=',', basefmt=' ')
plt.axhline(y=threshold_1, color='orange', linestyle='--', label=f'Threshold (4/n) = {threshold_1:.4f}')
plt.axhline(y=threshold_2, color='red', linestyle='--', label=f'Threshold (4/(n-k-1)) = {threshold_2:.4f}')
plt.xlabel('Observation Index', fontsize=12)
plt.ylabel("Cook's Distance", fontsize=12)
plt.title("Cook's Distance for All Observations", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Zoom in on potential outliers
plt.subplot(1, 2, 2)
if len(influential_points) > 0:
    # Show points above 50% of threshold
    zoom_threshold = influential_threshold * 0.5
    zoom_points = np.where(cooks_d > zoom_threshold)[0]
    
    plt.stem(zoom_points, cooks_d[zoom_points], markerfmt='o', basefmt=' ')
    plt.axhline(y=threshold_2, color='red', linestyle='--', label=f'Threshold = {threshold_2:.4f}')
    
    # Label top outliers
    top_n = min(5, len(influential_points))
    top_indices = influential_points[np.argsort(cooks_d[influential_points])[-top_n:]]
    for idx in top_indices:
        plt.annotate(f'{idx}', xy=(idx, cooks_d[idx]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Observation Index', fontsize=12)
    plt.ylabel("Cook's Distance", fontsize=12)
    plt.title("Influential Observations (Zoomed)", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'No influential\nobservations detected', 
             ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
    plt.axis('off')

plt.tight_layout()
plt.savefig("cooks_distance.png", dpi=150)
plt.show()
print("\nCook's distance plot saved as 'cooks_distance.png'")

# Option to refit without outliers
if len(influential_points) > 0:
    print("\n" + "="*60)
    print("REFITTING MODEL WITHOUT INFLUENTIAL OUTLIERS")
    print("="*60)
    
    # Create mask to exclude influential points
    non_influential_mask = np.ones(len(y), dtype=bool)
    non_influential_mask[influential_points] = False
    
    X_clean = X[non_influential_mask]
    y_clean = y[non_influential_mask]
    
    print(f"Original sample size: {len(y)}")
    print(f"Excluded observations: {len(influential_points)}")
    print(f"New sample size: {len(y_clean)}")
    
    # Fit cleaned model
    model_clean = sm.OLS(y_clean, X_clean).fit()
    
    print("\n--- MODEL WITHOUT OUTLIERS ---")
    print(f"R-squared:          {model_clean.rsquared:.6f} (was {model.rsquared:.6f})")
    print(f"Adj. R-squared:     {model_clean.rsquared_adj:.6f} (was {model.rsquared_adj:.6f})")
    print(f"AIC:                {model_clean.aic:.2f} (was {model.aic:.2f})")
    print(f"RMSE:               {np.sqrt(model_clean.mse_resid):.4f} (was {np.sqrt(model.mse_resid):.4f})")
    
    # Compare key coefficients
    print("\n--- COEFFICIENT COMPARISON ---")
    coef_comparison = pd.DataFrame({
        'Predictor': continuous_predictors,
        'Original_Coef': [model.params.get(pred, np.nan) for pred in continuous_predictors],
        'Clean_Coef': [model_clean.params.get(pred, np.nan) for pred in continuous_predictors],
    })
    coef_comparison['Difference'] = coef_comparison['Clean_Coef'] - coef_comparison['Original_Coef']
    coef_comparison['Pct_Change'] = (coef_comparison['Difference'] / coef_comparison['Original_Coef'].abs()) * 100
    
    print(coef_comparison.to_string(index=False))
    
    # Save cleaned model
    model_clean.save("immersion_ols_model_no_outliers.pickle")
    print("\nCleaned model saved as 'immersion_ols_model_no_outliers.pickle'")

model_mixed = smf.mixedlm(
    "immersion ~ mean_rot_speed_total_s + sd_rot_speed_total_s + total_angular_y_s + age_s + score_stai_t_s + vr_experience + C(video_type)",
    data=model_data,
    groups=model_data["participant"],
    re_formula="1"
).fit()

print(model_mixed.summary())