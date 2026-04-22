"""
Multi-group analysis by PHQ-9 severity (Minimal / Moderate / Severe)
for all 5 main DVs from h11.py.

Pipeline per DV:
  1. Shapiro-Wilk normality within each group
  2. Levene's test for homogeneity of variance
  3. Omnibus test:
       all normal + equal var  → One-way ANOVA
       all normal + unequal var → Welch ANOVA
       any non-normal           → Kruskal-Wallis
  4. If significant → post-hoc:
       ANOVA        → Tukey HSD
       Welch ANOVA  → Games-Howell  (pingouin)
       Kruskal-Wallis → Dunn's test (scikit-posthocs)
  5. Effect size:
       ANOVA / Welch → eta-squared  (SS_between / SS_total)
       Kruskal-Wallis → epsilon-squared
"""

import os
import numpy as np
import pandas as pd

from scipy.stats import shapiro, levene, f_oneway, kruskal
from pingouin import welch_anova, pairwise_gameshowell
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp


# =========================
# CONFIGURATION
# =========================
ALPHA = 0.05

METRIC_CONFIG = {
    "mean_speed": "Mean Speed (Total)",
    "sd_pitch": "SD - Pitch (X)",
    "sd_yaw": "SD - Yaw (Y)",
    "sd_roll": "SD - Roll (Z)",
    "total_movement": "Total Movement Magnitude",
}

VIDEO_CONFIG = {
    "v1": "V1",
    "v2": "V2",
    "v3": "V3",
    "v4": "V4",
    "v5": "V5",
}

SEVERITY_ORDER = ["Minimal", "Moderate", "Severe"]
BONFERRONI_ALPHA = ALPHA / len(METRIC_CONFIG)


# =========================
# UTILITIES
# =========================
def create_output_directory():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        "anova"
    )
    os.makedirs(path, exist_ok=True)
    return path


def print_section(title, char="="):
    print(f"\n{char*100}\n{title}\n{char*100}")


def interpret_effect_size(value):
    if np.isnan(value): return "—"
    if value >= 0.14: return "Large"
    if value >= 0.06: return "Medium"
    if value >= 0.01: return "Small"
    return "Negligible"


# =========================
# DATA PREPARATION
# =========================
def assign_severity_groups(df):
    def classify(phq):
        if phq <= 4: return "Minimal"
        if phq <= 10: return "Moderate"
        return "Severe"

    df["severity"] = df["score_phq"].apply(classify)
    df["severity"] = pd.Categorical(df["severity"], categories=SEVERITY_ORDER, ordered=True)
    return df


def compute_average_metrics(df):
    for metric_key, metric_label in METRIC_CONFIG.items():
        prefix = metric_label

        video_columns = [
            f"{prefix}_v{vid.replace('v','')}"
            for vid in VIDEO_CONFIG
            if f"{prefix}_v{vid.replace('v','')}" in df.columns
        ]

        df[f"avg_{metric_key}"] = df[video_columns].mean(axis=1)

    return df


# =========================
# EFFECT SIZE FUNCTIONS
# =========================
def eta_squared(groups):
    all_values = np.concatenate(groups)
    grand_mean = all_values.mean()

    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((all_values - grand_mean) ** 2)

    return ss_between / ss_total if ss_total > 0 else np.nan


def epsilon_squared(H, n):
    return (H - len(SEVERITY_ORDER) + 1) / (n - 1) if n > 1 else np.nan


# =========================
# CORE ANALYSIS
# =========================
def run_severity_analysis(df):
    results = []

    for metric_key, metric_label in METRIC_CONFIG.items():
        prefix = metric_label
        avg_column = f"avg_{metric_key}"

        print(f"\n{'-'*100}")
        print(f"DV: {metric_label}")
        print(f"{'-'*100}")

        # Prepare group data
        group_data = {
            severity: df.loc[df["severity"] == severity, avg_column].dropna().values
            for severity in SEVERITY_ORDER
        }

        valid_groups = [g for g in SEVERITY_ORDER if len(group_data[g]) > 0]
        group_arrays = [group_data[g] for g in valid_groups]

        # Descriptive stats
        for group in SEVERITY_ORDER:
            values = group_data[group]
            if len(values) > 0:
                print(f"  {group:10s}: n={len(values)}, M={values.mean():.3f}, SD={values.std(ddof=1):.3f}")
            else:
                print(f"  {group:10s}: n=0")

        # Normality
        all_normal = True
        print("\n  Shapiro-Wilk:")
        for group in valid_groups:
            values = group_data[group]

            if len(values) < 3:
                print(f"    {group}: n<3")
                all_normal = False
                continue

            _, p = shapiro(values)
            is_normal = p >= ALPHA
            if not is_normal:
                all_normal = False

            print(f"    {group}: p={p:.4f} {'OK' if is_normal else 'non-normal'}")

        # Levene
        if len(group_arrays) >= 2:
            _, lev_p = levene(*group_arrays)
            equal_var = lev_p >= ALPHA
        else:
            lev_p = np.nan
            equal_var = True

        # Select test
        if len(group_arrays) < 2:
            continue

        sub_df = df[["severity", avg_column]].dropna()

        if all_normal and equal_var:
            test_name = "One-way ANOVA"
            stat, p_val = f_oneway(*group_arrays)
            effect = eta_squared(group_arrays)

        elif all_normal:
            test_name = "Welch ANOVA"
            res = welch_anova(data=sub_df, dv=avg_column, between="severity")
            stat = res["F"].values[0]
            p_val = res["p-unc"].values[0]
            effect = eta_squared(group_arrays)

        else:
            test_name = "Kruskal-Wallis"
            stat, p_val = kruskal(*group_arrays)
            n_total = sum(len(g) for g in group_arrays)
            effect = epsilon_squared(stat, n_total)

        print(f"\n  {test_name}: p={p_val:.4f}, effect={effect:.4f} ({interpret_effect_size(effect)})")

        significant = p_val < ALPHA

        results.append({
            "Metric": metric_label,
            "Test": test_name,
            "p": round(p_val, 4),
            "Effect Size": round(effect, 4),
            "Interpretation": interpret_effect_size(effect),
            "Significant": "YES*" if significant else "no"
        })

        # Post-hoc
        if not significant:
            continue

        print("\n  POST-HOC:")

        if test_name == "One-way ANOVA":
            tukey = pairwise_tukeyhsd(sub_df[avg_column], sub_df["severity"])
            print(tukey.summary())

        elif test_name == "Welch ANOVA":
            gh = pairwise_gameshowell(data=sub_df, dv=avg_column, between="severity")
            print(gh[["A", "B", "pval"]].to_string(index=False))

        else:
            dunn = sp.posthoc_dunn(sub_df, val_col=avg_column, group_col="severity", p_adjust="bonferroni")
            print(dunn.round(4).to_string())

    return pd.DataFrame(results)


# =========================
# SUMMARY
# =========================
def summarize_results(results_df, output_dir):
    print_section("SUMMARY")

    print(results_df.to_string(index=False))

    significant = results_df[results_df["Significant"] == "YES*"]
    print(f"\nSignificant: {len(significant)} / {len(results_df)}")

    output_path = os.path.join(output_dir, "severity_omnibus_results.csv")
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved -> {output_path}")


# =========================
# MAIN
# =========================
def main():
    df = pd.read_csv("output.csv")

    output_dir = create_output_directory()

    df = assign_severity_groups(df)
    df = compute_average_metrics(df)

    results_df = run_severity_analysis(df)
    summarize_results(results_df, output_dir)


if __name__ == "__main__":
    main()