"""
PRELIMINARY ANALYSIS:
- This file is used to perform the preliminary analysis on the data including:
        - correlation matrix
        - normality testing
        - group comparisons using Welch's t-test or Mann-Whitney U test
        - Cohen's d for effect size
        - summary tables for all group comparisons
        - saving the output to a CSV file
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu


# =========================
# CONFIGURATION
# =========================
ALPHA = 0.05

METRIC_CONFIG = {
    "mean_speed": {
        "label": "Mean Speed (deg/s)",
        "column_prefix": "Mean Speed (Total)",
    },
    "sd_pitch": {
        "label": "SD Pitch (deg)",
        "column_prefix": "SD - Pitch (X)",
    },
    "sd_yaw": {
        "label": "SD Yaw (deg)",
        "column_prefix": "SD - Yaw (Y)",
    },
    "sd_roll": {
        "label": "SD Roll (deg)",
        "column_prefix": "SD - Roll (Z)",
    },
    "total_movement": {
        "label": "Total Movement Magnitude",
        "column_prefix": "Total Movement Magnitude",
    },
}

VIDEO_CONFIG = {
    "v1": "V1: Abandoned",
    "v2": "V2: Beach",
    "v3": "V3: Campus",
    "v4": "V4: Horror",
    "v5": "V5: Surf",
}


# =========================
# UTILITIES
# =========================
def create_output_directory():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        "preliminary_analysis"
    )
    os.makedirs(path, exist_ok=True)
    return path


def print_section(title, char="="):
    print(f"\n{char*100}\n{title}\n{char*100}")


def save_figure(fig, filepath):
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def interpret_cohens_d(effect_size):
    abs_d = abs(effect_size)
    if abs_d >= 0.8:
        return "Large"
    elif abs_d >= 0.5:
        return "Medium"
    elif abs_d >= 0.2:
        return "Small"
    return "Negligible"


# =========================
# DATA PREPARATION
# =========================
def compute_average_metrics(df):
    """Compute average metric across videos."""
    for metric_key, metric_info in METRIC_CONFIG.items():
        prefix = metric_info["column_prefix"]
        matching_cols = df.filter(regex=f"^{prefix}_v").columns
        
        if len(matching_cols) > 0:
            df[f"avg_{metric_key}"] = df[matching_cols].mean(axis=1)
    return df


# =========================
# CORRELATION MATRIX
# =========================
def correlation_analysis(df, output_dir):
    print_section("CORRELATION MATRIX", "-")

    df = compute_average_metrics(df)

    variable_labels = {
        "score_phq": "PHQ-9",
        "score_gad": "GAD-7",
        "score_stai_t": "STAI-T",
        "score_vrise": "VRISE",
        "positive_affect_start": "PANAS+",
        "negative_affect_start": "PANAS-",
        "avg_mean_speed": "Avg Speed",
        "avg_sd_yaw": "Avg SD Yaw",
        "avg_sd_pitch": "Avg SD Pitch",
        "avg_sd_roll": "Avg SD Roll",
    }

    valid_columns = [col for col in variable_labels if col in df.columns]
    correlation_matrix = df[valid_columns].corr()
    correlation_matrix.rename(index=variable_labels, columns=variable_labels, inplace=True)

    print("\nCorrelation matrix:")
    print(correlation_matrix.round(3).to_string())

    fig, ax = plt.subplots(figsize=(9, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="BrBG",
        center=0,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 8},
        ax=ax
    )

    ax.set_title("Correlation Matrix: Psychological & Headtracking Measures", fontweight='bold')

    save_figure(fig, os.path.join(output_dir, "Correlation_Matrix.png"))


# =========================
# NORMALITY TESTING
# =========================
def perform_normality_tests(df):
    print_section("NORMALITY TESTING (Shapiro-Wilk)")

    results = []

    for metric_key, metric_info in METRIC_CONFIG.items():
        column_prefix = metric_info["column_prefix"]
        metric_label = metric_info["label"]

        for video_id, video_label in VIDEO_CONFIG.items():
            column_name = f"{column_prefix}_{video_id}"

            if column_name not in df.columns:
                continue

            for group_value, group_label in [(0, "Non-Depressed"), (1, "Depressed")]:
                sample = df.loc[df["phq_group"] == group_value, column_name].dropna()

                if len(sample) < 3:
                    results.append({
                        "Metric": metric_label,
                        "Video": video_label,
                        "Group": group_label,
                        "n": len(sample),
                        "W": np.nan,
                        "p": np.nan,
                        "Normal": "n<3",
                    })
                    continue

                stat, p_value = shapiro(sample)

                results.append({
                    "Metric": metric_label,
                    "Video": video_label,
                    "Group": group_label,
                    "n": len(sample),
                    "W": round(stat, 4),
                    "p": round(p_value, 4),
                    "Normal": "YES" if p_value >= ALPHA else "NO*",
                })

    normality_df = pd.DataFrame(results)

    print(normality_df.to_string(index=False))

    violations = normality_df[normality_df["Normal"] == "NO*"]
    print(f"\nViolations (p < {ALPHA}): {len(violations)} / {len(normality_df)}")

    return normality_df


# =========================
# GROUP COMPARISON
# =========================
def perform_group_comparisons(df, normality_df):
    print_section("GROUP COMPARISONS")

    normality_lookup = {
        (row["Metric"], row["Video"], row["Group"]): (row["Normal"] == "YES")
        for _, row in normality_df.iterrows()
    }

    comparison_results = []

    for metric_key, metric_info in METRIC_CONFIG.items():
        column_prefix = metric_info["column_prefix"]
        metric_label = metric_info["label"]

        print(f"\n== {metric_label} ==")

        for video_id, video_label in VIDEO_CONFIG.items():
            column_name = f"{column_prefix}_{video_id}"

            if column_name not in df.columns:
                continue

            video_label = video_label

            group_nd = df.loc[df["phq_group"] == 0, column_name].dropna().values
            group_d = df.loc[df["phq_group"] == 1, column_name].dropna().values

            if len(group_nd) < 3 or len(group_d) < 3:
                print(f"  {video_label}: SKIPPED")
                continue

            # Normality
            nd_normal = normality_lookup.get((metric_label, video_label, "Non-Depressed"), False)
            d_normal = normality_lookup.get((metric_label, video_label, "Depressed"), False)

            # Levene
            _, levene_p = levene(group_nd, group_d)
            equal_variance = levene_p >= ALPHA

            # Test selection
            if nd_normal and d_normal:
                stat, p_value = ttest_ind(group_nd, group_d, equal_var=False)
                test_name = "Welch t"
            else:
                stat, p_value = mannwhitneyu(group_nd, group_d)
                test_name = "Mann-Whitney"

            # Effect size
            pooled_sd = np.sqrt(
                ((len(group_nd)-1)*np.var(group_nd, ddof=1) +
                 (len(group_d)-1)*np.var(group_d, ddof=1)) /
                (len(group_nd) + len(group_d) - 2)
            )

            effect_size = (
                (np.mean(group_nd) - np.mean(group_d)) / pooled_sd
                if pooled_sd > 0 else np.nan
            )

            comparison_results.append({
                "Metric": metric_label,
                "Video": video_label,
                "Test": test_name,
                "p": round(p_value, 4),
                "Cohen_d": round(effect_size, 3),
                "Effect": interpret_cohens_d(effect_size),
            })

            print(f"  {video_label}: {test_name} p={p_value:.4f}, d={effect_size:.3f}")

    return pd.DataFrame(comparison_results)


# =========================
# SUMMARY
# =========================
def summarize_results(results_df):
    print_section("SUMMARY")

    print("\nFull Results:")
    print(results_df.to_string(index=False))

    significant = results_df[results_df["p"] < ALPHA]

    print(f"\nSignificant results: {len(significant)} / {len(results_df)}")

    print("\nPivot (p-values):")
    print(results_df.pivot_table(index="Metric", columns="Video", values="p").round(4))

    print("\nPivot (Cohen's d):")
    print(results_df.pivot_table(index="Metric", columns="Video", values="Cohen_d").round(3))


# =========================
# MAIN
# =========================
def main():
    df = pd.read_csv("output.csv")
    df.drop(columns=["phq_label"], inplace=True, errors="ignore")

    output_dir = create_output_directory()

    correlation_analysis(df, output_dir)
    normality_df = perform_normality_tests(df)
    results_df = perform_group_comparisons(df, normality_df)
    summarize_results(results_df)


if __name__ == "__main__":
    main()