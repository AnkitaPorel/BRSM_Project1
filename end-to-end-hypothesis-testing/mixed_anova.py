"""
Mixed ANOVA: PHQ-9 Severity (between) × Video (within)

Design
------
  Between-subjects : severity  (Minimal ≤4 | Moderate 5-10 | Severe 11+)
  Within-subjects  : video     (V1-V5, all 5 videos)
  Subject          : participant
  DVs (5)          : mean_speed, sd_pitch, sd_yaw, sd_roll, total_movement

Pipeline per DV
---------------
  1. Reshape to long  (1 row per participant × video)
  2. Assumption checks
       a. Normality of residuals  (Shapiro-Wilk)
       b. Homogeneity of between-group variance  (Levene)
       c. Sphericity  (handled by pingouin — Greenhouse-Geisser auto-applied)
  3. Mixed ANOVA  → main effect severity, main effect video, interaction
  4. Post-hoc (if significant)
       Between effect  → pairwise t-tests across severity groups
       Within effect   → pairwise t-tests across videos
       Interaction     → simple effects (compare groups within each video)
  5. Effect sizes  → partial η² reported by pingouin for each term
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro, levene
from pingouin import mixed_anova, pairwise_tests

warnings.filterwarnings("ignore")


# =========================
# CONFIGURATION
# =========================
ALPHA = 0.05

METRIC_CONFIG = {
    "mean_speed": {
        "column_prefix": "Mean Speed (Total)",
        "label": "Mean Speed (deg/s)",
    },
    "sd_pitch": {
        "column_prefix": "SD - Pitch (X)",
        "label": "SD Pitch (deg)",
    },
    "sd_yaw": {
        "column_prefix": "SD - Yaw (Y)",
        "label": "SD Yaw (deg)",
    },
    "sd_roll": {
        "column_prefix": "SD - Roll (Z)",
        "label": "SD Roll (deg)",
    },
    "total_movement": {
        "column_prefix": "Total Movement Magnitude",
        "label": "Total Movement",
    },
}

VIDEO_CONFIG = {
    "v1": "V1: Abandoned",
    "v2": "V2: Beach",
    "v3": "V3: Campus",
    "v4": "V4: Horror",
    "v5": "V5: Surf",
}

SEVERITY_ORDER = ["Minimal", "Moderate", "Severe"]

PALETTE = {
    "Minimal": "steelblue",
    "Moderate": "orange",
    "Severe": "tomato",
}


# =========================
# UTILITIES
# =========================
def create_output_directory():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        "mixed_anova"
    )
    os.makedirs(path, exist_ok=True)
    return path


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def interpret_eta_squared(value):
    if np.isnan(value): return "-"
    if value >= 0.14: return "Large"
    if value >= 0.06: return "Medium"
    if value >= 0.01: return "Small"
    return "Negligible"


# =========================
# DATA PREPARATION
# =========================
def assign_severity(df):
    def classify(phq):
        if phq <= 4: return "Minimal"
        if phq <= 10: return "Moderate"
        return "Severe"

    df["severity"] = pd.Categorical(
        df["score_phq"].apply(classify),
        categories=SEVERITY_ORDER,
        ordered=True
    )
    return df


def reshape_to_long(df):
    records = []

    for _, row in df.iterrows():
        for video_id, video_label in VIDEO_CONFIG.items():
            video_index = video_id.replace("v", "")

            base_record = {
                "participant": row["participant"],
                "severity": row["severity"],
                "video": video_label,
            }

            for metric_key, metric_info in METRIC_CONFIG.items():
                col_name = f"{metric_info['column_prefix']}_v{video_index}"
                base_record[metric_key] = row[col_name] if col_name in df.columns else np.nan

            records.append(base_record)

    long_df = pd.DataFrame(records)

    long_df["severity"] = pd.Categorical(
        long_df["severity"], categories=SEVERITY_ORDER, ordered=True
    )

    long_df["video"] = pd.Categorical(
        long_df["video"],
        categories=list(VIDEO_CONFIG.values()),
        ordered=True
    )

    return long_df.dropna(subset=list(METRIC_CONFIG.keys()), how="all")


def filter_complete_cases(df, dv):
    counts = df.groupby("participant")["video"].nunique()
    complete_ids = counts[counts == len(VIDEO_CONFIG)].index
    return df[df["participant"].isin(complete_ids)][
        ["participant", "severity", "video", dv]
    ].dropna()


# =========================
# ANALYSIS CORE
# =========================
def run_mixed_anova(df_long):
    summary_rows = []

    for metric_key, metric_info in METRIC_CONFIG.items():
        dv = metric_key
        label = metric_info["label"]

        sub_df = filter_complete_cases(df_long, dv)

        print_section(f"DV: {label} (n={sub_df['participant'].nunique()})")

        # Descriptives
        descriptives = (
            sub_df.groupby(["severity", "video"], observed=True)[dv]
            .agg(mean="mean", sd="std", n="count")
            .reset_index()
        )
        print("\nDescriptives:")
        print(descriptives.to_string(index=False))

        # Residual normality
        cell_means = sub_df.groupby(["severity", "video"], observed=True)[dv].transform("mean")
        residuals = sub_df[dv] - cell_means

        if len(residuals) >= 3:
            _, p_norm = shapiro(residuals)
            normal = p_norm >= ALPHA
        else:
            p_norm, normal = np.nan, False

        print(f"\nResidual normality p={p_norm:.4f} -> {'OK' if normal else 'Non-normal'}")

        # Levene
        groups = [g[dv].values for _, g in sub_df.groupby("severity", observed=True)]
        if len(groups) >= 2:
            _, lev_p = levene(*groups)
            equal_var = lev_p >= ALPHA
        else:
            lev_p, equal_var = np.nan, True

        print(f"Levene p={lev_p:.4f} -> {'Equal var' if equal_var else 'Unequal var'}")

        # Mixed ANOVA
        try:
            aov = mixed_anova(
                data=sub_df,
                dv=dv,
                within="video",
                between="severity",
                subject="participant"
            )
        except Exception as e:
            print(f"Mixed ANOVA failed: {e}")
            continue

        print("\nANOVA Results:")
        print(aov.to_string(index=False))

        # Store results
        for _, row in aov.iterrows():
            summary_rows.append({
                "DV": label,
                "Source": row["Source"],
                "F": round(row["F"], 4),
                "p": round(row["p-unc"], 4),
                "np2": round(row["np2"], 4),
                "Effect": interpret_eta_squared(row["np2"]),
                "Sig": "YES*" if row["p-unc"] < ALPHA else "no",
            })

        # Post-hoc
        run_posthoc(sub_df, dv, aov, normal)

    return pd.DataFrame(summary_rows)


def run_posthoc(df, dv, aov, is_parametric):
    def print_posthoc(result, cols):
        cols = [c for c in cols if c in result.columns]
        print(result[cols].to_string(index=False))

    # Between
    if "severity" in aov["Source"].values:
        p = aov[aov["Source"] == "severity"]["p-unc"].values[0]
        if p < ALPHA:
            print("\nPOST-HOC (Between):")
            res = pairwise_tests(df, dv=dv, between="severity",
                                 subject="participant", padjust="bonf",
                                 parametric=is_parametric)
            print_posthoc(res, ["A", "B", "p-corr", "hedges"])

    # Within
    if "video" in aov["Source"].values:
        p = aov[aov["Source"] == "video"]["p-unc"].values[0]
        if p < ALPHA:
            print("\nPOST-HOC (Within):")
            res = pairwise_tests(df, dv=dv, within="video",
                                 subject="participant", padjust="bonf",
                                 parametric=is_parametric)
            print_posthoc(res, ["A", "B", "p-corr", "hedges"])

    # Interaction (Source name from pingouin is "Interaction")
    interaction_row = aov[aov["Source"] == "Interaction"]
    if len(interaction_row) and interaction_row["p-unc"].values[0] < ALPHA:
        print("\nPOST-HOC (Interaction):")
        res = pairwise_tests(df, dv=dv, within="video",
                             between="severity", subject="participant",
                             padjust="bonf", parametric=is_parametric)
        print_posthoc(res, ["video", "A", "B", "p-corr"])


# =========================
# PLOTTING
# =========================
def generate_interaction_plots(df_long, output_dir):
    print("\nGenerating interaction plots...")

    for metric_key, metric_info in METRIC_CONFIG.items():
        dv = metric_key
        label = metric_info["label"]

        sub_df = filter_complete_cases(df_long, dv)

        interaction = (
            sub_df.groupby(["video", "severity"], observed=True)[dv]
            .agg(mean="mean", sd="std", n="count")
            .reset_index()
        )
        interaction["se"] = interaction["sd"] / np.sqrt(interaction["n"])

        fig, ax = plt.subplots(figsize=(10, 5))

        for group in SEVERITY_ORDER:
            subset = interaction[interaction["severity"] == group]
            ax.errorbar(
                subset["video"].astype(str),
                subset["mean"],
                yerr=subset["se"],
                marker="o",
                label=group,
                color=PALETTE[group],
                capsize=4
            )

        ax.set_title(f"{label}: Severity x Video", fontweight="bold")
        ax.set_ylabel(label)
        ax.legend()
        ax.tick_params(axis="x", rotation=15)

        fig.tight_layout()
        path = os.path.join(output_dir, f"interaction_{metric_key}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)

        print(f"Saved: {path}")


# =========================
# SUMMARY
# =========================
def summarize_results(df, output_dir):
    print_section("SUMMARY")
    print(df.to_string(index=False))

    path = os.path.join(output_dir, "mixed_anova_summary.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved -> {path}")


# =========================
# MAIN
# =========================
def main():
    df = pd.read_csv("output.csv")
    output_dir = create_output_directory()

    df = assign_severity(df)
    df_long = reshape_to_long(df)

    print(f"\nLong data shape: {df_long.shape}")

    summary = run_mixed_anova(df_long)
    summarize_results(summary, output_dir)

    generate_interaction_plots(df_long, output_dir)

    print("\nMixed ANOVA analysis complete.")


if __name__ == "__main__":
    main()