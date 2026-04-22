"""
# Non-parametric within-subject video analysis

#   1. Friedman test — repeated measures across 5 videos (per DV)
#   2. If significant: Bonferroni-corrected Wilcoxon signed-rank pairwise post-hocs
#   3. Kruskal–Wallis — emotional category (Positive / Neutral / Negative) on long-format data

"""

import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, kruskal

warnings.filterwarnings("ignore")


# =========================
# CONFIGURATION
# =========================
ALPHA = 0.05

VIDEO_CONFIG = {
    "v1": "V1: Abandoned",
    "v2": "V2: Beach",
    "v3": "V3: Campus",
    "v4": "V4: Horror",
    "v5": "V5: Surf",
}

METRIC_CONFIG = {
    "mean_speed": {
        "prefix": "Mean Speed (Total)",
        "label": "Mean Speed (deg/s)",
    },
    "sd_pitch": {
        "prefix": "SD - Pitch (X)",
        "label": "SD Pitch (deg)",
    },
    "sd_yaw": {
        "prefix": "SD - Yaw (Y)",
        "label": "SD Yaw (deg)",
    },
    "sd_roll": {
        "prefix": "SD - Roll (Z)",
        "label": "SD Roll (deg)",
    },
    "total_movement": {
        "prefix": "Total Movement Magnitude",
        "label": "Total Movement",
    },
}

CATEGORY_CONFIG = {
    "Positive": ["v2", "v5"],
    "Neutral": ["v3"],
    "Negative": ["v1", "v4"],
}

CATEGORY_ORDER = ["Positive", "Neutral", "Negative"]


# =========================
# UTILITIES
# =========================
def create_output_directory():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        "non_param"
    )
    os.makedirs(path, exist_ok=True)
    return path


def kendalls_w(chi2, k, n):
    return chi2 / (n * (k - 1)) if n > 0 and k > 1 else np.nan


def epsilon_squared(H, k, n):
    return (H - k + 1) / (n - k + 1) if n > 1 else np.nan


# =========================
# DATA TRANSFORMATION
# =========================
def build_long_format(df):
    records = []

    for _, row in df.iterrows():
        for vid, vid_label in VIDEO_CONFIG.items():
            vid_num = vid.replace("v", "")

            base = {
                "participant": row["participant"],
                "video": vid,
                "video_label": vid_label,
                "video_category": next(
                    cat for cat, vids in CATEGORY_CONFIG.items() if vid in vids
                ),
            }

            for metric_key, metric_info in METRIC_CONFIG.items():
                col = f"{metric_info['prefix']}_v{vid_num}"
                base[metric_key] = row[col] if col in df.columns else np.nan

            records.append(base)

    long_df = pd.DataFrame(records)

    long_df["video_label"] = pd.Categorical(
        long_df["video_label"],
        categories=list(VIDEO_CONFIG.values()),
        ordered=True
    )

    long_df["video_category"] = pd.Categorical(
        long_df["video_category"],
        categories=CATEGORY_ORDER,
        ordered=True
    )

    return long_df


# =========================
# FRIEDMAN + WILCOXON
# =========================
def run_friedman_analysis(df):
    friedman_results = []
    wilcoxon_results = []

    for metric_key, metric_info in METRIC_CONFIG.items():
        label = metric_info["label"]
        prefix = metric_info["prefix"]

        cols = [
            f"{prefix}_v{vid.replace('v','')}"
            for vid in VIDEO_CONFIG
            if f"{prefix}_v{vid.replace('v','')}" in df.columns
        ]

        if len(cols) < len(VIDEO_CONFIG):
            continue

        wide_df = df[["participant"] + cols].dropna()
        n = len(wide_df)

        if n < 3:
            continue

        matrices = [wide_df[c].values for c in cols]
        chi2, p_val = friedmanchisquare(*matrices)
        w = kendalls_w(chi2, len(VIDEO_CONFIG), n)

        print(f"\n{label}")
        print(f"  Friedman: chi2={chi2:.4f}, p={p_val:.6f}, W={w:.4f}")

        friedman_results.append({
            "DV": label,
            "chi2": chi2,
            "p": p_val,
            "kendall_W": w,
        })

        if p_val >= ALPHA:
            continue

        pairs = list(combinations(range(len(VIDEO_CONFIG)), 2))
        raw_p = []
        stats_vals = []

        for i, j in pairs:
            try:
                stat, p = wilcoxon(matrices[i], matrices[j])
            except:
                stat, p = np.nan, 1.0

            raw_p.append(p)
            stats_vals.append(stat)

        p_adj = np.minimum(np.array(raw_p) * len(pairs), 1.0)

        for idx, (i, j) in enumerate(pairs):
            vi = list(VIDEO_CONFIG.values())[i]
            vj = list(VIDEO_CONFIG.values())[j]

            wilcoxon_results.append({
                "DV": label,
                "A": vi,
                "B": vj,
                "p_raw": raw_p[idx],
                "p_adj": p_adj[idx],
                "sig": p_adj[idx] < ALPHA,
            })

    return pd.DataFrame(friedman_results), pd.DataFrame(wilcoxon_results)


# =========================
# KRUSKAL-WALLIS
# =========================
def run_kruskal_analysis(long_df):
    results = []

    for metric_key, metric_info in METRIC_CONFIG.items():
        label = metric_info["label"]

        sub = long_df[["video_category", metric_key]].dropna()

        groups = [
            g[metric_key].values
            for _, g in sub.groupby("video_category", observed=True)
        ]

        if len(groups) < 2:
            continue

        H, p = kruskal(*groups)
        n = len(sub)
        k = len(groups)
        eps = epsilon_squared(H, k, n)

        print(f"\n{label}")
        print(f"  Kruskal-Wallis: H={H:.4f}, p={p:.6f}, eps2={eps:.4f}")

        results.append({
            "DV": label,
            "H": H,
            "p": p,
            "epsilon2": eps,
        })

    return pd.DataFrame(results)


# =========================
# MAIN
# =========================
def main():
    df = pd.read_csv("output.csv")
    output_dir = create_output_directory()

    long_df = build_long_format(df)

    print("\nRunning Friedman + Wilcoxon...")
    friedman_df, wilcoxon_df = run_friedman_analysis(df)

    print("\nRunning Kruskal-Wallis...")
    kruskal_df = run_kruskal_analysis(long_df)

    friedman_df.to_csv(os.path.join(output_dir, "friedman.csv"), index=False)
    wilcoxon_df.to_csv(os.path.join(output_dir, "wilcoxon.csv"), index=False)
    kruskal_df.to_csv(os.path.join(output_dir, "kruskal.csv"), index=False)

    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()

