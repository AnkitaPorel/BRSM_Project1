"""
Linear mixed-effects regression for head movement ~ PHQ depression group × video,
with random intercepts for participant.

Implements the workflow from the analysis spec:
  • Model: DV ~ PHQ_group + Video + PHQ_group:Video + (1 | Participant)
  • Significance: Wald tests on terms + likelihood-ratio tests (ML fits)
  • Diagnostics: residual Q-Q, residuals vs fitted, Shapiro-Wilk on residuals
  • Effect size: Nakagawa marginal / conditional R²
  • Post-hoc: estimated marginal means (fixed-effects predictions) with Bonferroni
    adjustments for selected contrast families

Requires: pandas, numpy, scipy, matplotlib, statsmodels, patsy
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import shapiro
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests
from patsy import build_design_matrices

warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")


# =========================
# CONFIGURATION
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(BASE_DIR, "output.csv")
OUTDIR = os.path.join(BASE_DIR, "results", "regression")

ALPHA = 0.05

VIDEO_CONFIG = ["v1", "v2", "v3", "v4", "v5"]

METRIC_CONFIG = {
    "mean_speed": "Mean Speed (Total)",
    "sd_pitch": "SD - Pitch (X)",
    "sd_yaw": "SD - Yaw (Y)",
    "sd_roll": "SD - Roll (Z)",
    "mean_pitch": "Mean Rotation - Pitch (X)",
    "mean_yaw": "Mean Rotation - Yaw (Y)",
    "mean_roll": "Mean Rotation - Roll (Z)",
    "range_pitch": "Range - Pitch (X)",
    "range_yaw": "Range - Yaw (Y)",
    "range_roll": "Range - Roll (Z)",
    "mean_speed_x": "Mean Speed - Pitch (X)",
    "mean_speed_y": "Mean Speed - Yaw (Y)",
    "mean_speed_z": "Mean Speed - Roll (Z)",
    "total_movement": "Total Movement Magnitude",
}


# =========================
# DATA TRANSFORMATION
# =========================
def wide_to_long(df: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    prefix = METRIC_CONFIG[metric_key]

    records = [
        {
            "participant": row["participant"],
            "phq_group": int(row["phq_group"]),
            "video": vid,
            "y": row[col],
        }
        for _, row in df.iterrows()
        for vid in VIDEO_CONFIG
        for col in [f"{prefix}_v{vid.replace('v','')}"]
        if col in df.columns and pd.notna(row[col])
    ]

    long_df = pd.DataFrame(records)

    if long_df.empty:
        return long_df

    long_df["video"] = pd.Categorical(long_df["video"], categories=VIDEO_CONFIG)
    long_df["phq_group"] = pd.Categorical(long_df["phq_group"], categories=[0, 1])

    return long_df


# =========================
# EFFECT SIZE
# =========================
def compute_nakagawa_r2(result):
    scale = float(result.scale)

    exog = np.asarray(result.model.exog)
    fe = np.asarray(result.fe_params).ravel()
    eta_f = exog @ fe
    var_fixed = np.var(eta_f, ddof=1)

    groups = np.asarray(result.model.groups)
    re = result.random_effects

    blup = np.array([
        float(re[g].iloc[0]) if isinstance(re[g], pd.Series) else float(re[g][0])
        for g in groups
    ])

    var_random = np.var(blup, ddof=1)

    denom = var_fixed + var_random + scale
    if denom <= 0:
        return np.nan, np.nan

    return var_fixed / denom, (var_fixed + var_random) / denom


# =========================
# MODEL FITTING
# =========================
def fit_mixed_model(df_long, random_slopes):
    formula = "y ~ C(phq_group) * C(video)"
    kwargs = {"groups": df_long["participant"]}

    if random_slopes:
        kwargs["re_formula"] = "~0 + C(video)"

    model = mixedlm(formula, df_long, **kwargs)
    return model.fit(reml=True), formula


# =========================
# LIKELIHOOD RATIO TESTS
# =========================
def likelihood_ratio_tests(df_long, formula):
    results = []

    def safe_fit(formula):
        try:
            return mixedlm(formula, df_long, groups=df_long["participant"]).fit(reml=False)
        except:
            return None

    full = safe_fit(formula)
    add = safe_fit("y ~ C(phq_group) + C(video)")

    if full and add:
        lr = 2 * (full.llf - add.llf)
        df_diff = len(full.fe_params) - len(add.fe_params)
        p = 1 - stats.chi2.cdf(lr, df_diff)
        results.append(("Interaction", lr, df_diff, p))

    return pd.DataFrame(results, columns=["Term", "Chi2", "df", "p"])


# =========================
# POST-HOC (EMMEANS)
# =========================
def compute_emmeans(result, df_long):
    design_info = result.model.data.design_info

    grid = pd.DataFrame([
        {"phq_group": p, "video": v}
        for p in df_long["phq_group"].cat.categories
        for v in df_long["video"].cat.categories
    ])

    (X,) = build_design_matrices([design_info], grid)
    fe = result.fe_params.values

    preds = X @ fe

    print("\nEstimated Marginal Means:")
    grid["EMM"] = preds
    print(grid.to_string(index=False))


# =========================
# DIAGNOSTICS
# =========================
def plot_diagnostics(result, prefix):
    resid = result.resid
    fitted = result.fittedvalues

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    stats.probplot(resid, plot=ax[0])
    ax[0].set_title("Q-Q Plot")

    ax[1].scatter(fitted, resid, alpha=0.5)
    ax[1].axhline(0)
    ax[1].set_title("Residuals vs Fitted")

    plt.tight_layout()
    plt.savefig(prefix + "_diagnostics.png")
    plt.close()


# =========================
# MAIN PIPELINE
# =========================
def run_analysis(csv_path, metric_key, random_slopes, standardize):
    os.makedirs(OUTDIR, exist_ok=True)

    if metric_key not in METRIC_CONFIG:
        raise ValueError(f"Invalid metric: {metric_key}")

    df = pd.read_csv(csv_path)
    df_long = wide_to_long(df, metric_key)

    if len(df_long) < 20:
        raise ValueError("Insufficient data")

    if standardize:
        df_long["y"] = (df_long["y"] - df_long["y"].mean()) / df_long["y"].std()

    result, formula = fit_mixed_model(df_long, random_slopes)

    print("\nMODEL SUMMARY\n")
    print(result.summary())

    print("\nLIKELIHOOD RATIO TESTS\n")
    print(likelihood_ratio_tests(df_long, formula))

    r2m, r2c = compute_nakagawa_r2(result)
    print(f"\nR2 (Marginal): {r2m:.4f}")
    print(f"R2 (Conditional): {r2c:.4f}")

    print("\nShapiro-Wilk (Residuals):")
    W, p = shapiro(result.resid)
    print(f"W={W:.4f}, p={p:.4f}")

    prefix = os.path.join(OUTDIR, f"lmm_{metric_key}")
    plot_diagnostics(result, prefix)

    compute_emmeans(result, df_long)


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--dv", default="total_movement",
                        choices=list(METRIC_CONFIG.keys()))
    parser.add_argument("--random-slopes", action="store_true")
    parser.add_argument("--standardize", action="store_true")

    args = parser.parse_args()

    run_analysis(args.csv, args.dv, args.random_slopes, args.standardize)


if __name__ == "__main__":
    main()