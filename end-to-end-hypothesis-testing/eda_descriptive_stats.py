""" 
══════════════════════════════════════════════════════════════════════════════
STEP 2:DESCRIPTIVE STATISTICS:
- This file is used to generate the descriptive statistics from the updated dataset(output.csv) including:
        - demographic characteristics
        - clinical characteristics
        - psychological scales
        - valence & arousal by video
        - headtracking descriptives
═══════════════════════════════════════════════════════════════════════════════
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
SCALES = [
    ("PHQ-9", "score_phq"),
    ("GAD-7", "score_gad"),
    ("STAI-T", "score_stai_t"),
    ("VRISE", "score_vrise"),
    ("PANAS +ve (Pre)", "positive_affect_start"),
    ("PANAS -ve (Pre)", "negative_affect_start"),
    ("PANAS +ve (Post)", "positive_affect_end"),
    ("PANAS -ve (Post)", "negative_affect_end"),
]

VIDEO_LABELS = {
    "v1": "V1: Abandoned",
    "v2": "V2: Beach",
    "v3": "V3: Campus",
    "v4": "V4: Horror",
    "v5": "V5: Surf",
}

ALL_VIDEOS = ["v1", "v2", "v3", "v4", "v5"]

FIG1_CFG = [
    ("PHQ-9 (Depression)", "score_phq", "rebeccapurple", 2),
    ("GAD-7 (Anxiety)", "score_gad", "royalblue", 2),
    ("STAI-T (Trait Anxiety)", "score_stai_t", "deeppink", 5),
    ("VRISE (Simulator Sickness)", "score_vrise", "forestgreen", 2),
]

FIG4_CFG = [
    ("mean_speed", "Mean Rotation Speed by Video", "Mean Speed (°/s)", "skyblue"),
    ("sd_yaw", "SD of Yaw by Video", "SD Yaw (°)", "lightsalmon"),
    ("sd_pitch", "SD of Pitch by Video", "SD Pitch (°)", "lightgreen"),
    ("sd_roll", "SD of Roll by Video", "SD Roll (°)", "plum"),
]


# =========================
# UTILITIES
# =========================
def create_output_dir():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "eda_descriptive_stats")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def print_header(title):
    print(f"\n{'-'*100}\n{title}\n{'-'*100}\n")


def print_section(title):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70 + "\n")


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# =========================
# DESCRIPTIVE STATS
# =========================
def demographic_stats(df):
    print_section("DEMOGRAPHIC SUMMARY STATISTICS")

    print("N =", len(df))
    print(f"Age: M={df['age'].mean():.2f}, SD={df['age'].std():.2f}, "
          f"Range={df['age'].min()}-{df['age'].max()}")
    print("Gender:\n", df['gender'].value_counts())
    print("VR Experience:\n", df['vr_experience'].value_counts())


def clinical_stats(df):
    print_section("CLINICAL SUMMARY STATISTICS")

    rows = []
    for label, col in SCALES:
        s = df[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])

        rows.append({
            "Scale": label,
            "Mean": round(s.mean(), 2),
            "Median": round(s.median(), 1),
            "SD": round(s.std(), 2),
            "Range": f"[{int(s.min())}, {int(s.max())}]",
        })

    table = pd.DataFrame(rows).set_index("Scale")
    print(table.to_string())

    print("\nPHQ Groups:")
    for group, count in df["phq_group"].value_counts().items():
        label = "Low(<10)" if group == 0 else "High(>=10)"
        print(f"{label} ({group}): {count}")


# =========================
# PLOTTING FUNCTIONS
# =========================
def plot_histograms(df, outdir):
    print_section("PSYCHOLOGICAL SCALES SUMMARY STATISTICS")

    fig, axes = plt.subplots(1, len(FIG1_CFG), figsize=(16, 4))

    for ax, (title, col, color, bw) in zip(axes, FIG1_CFG):
        s = df[col].dropna()
        bins = np.arange(s.min(), s.max() + bw + 1, bw)

        ax.hist(s, bins=bins, color=color, edgecolor='white', alpha=0.8)
        ax.set(title=title, xlabel="Score", ylabel="Count")

    save_fig(fig, os.path.join(outdir, "psychological_scales_summary.png"))


def build_valence_arousal_long(df):
    rows = [
        {
            "participant": r["participant"],
            "video": VIDEO_LABELS[v],
            "Valence": r[f"valence_{v}"],
            "Arousal": r[f"arousal_{v}"],
        }
        for v in ALL_VIDEOS
        for _, r in df.iterrows()
    ]

    va = pd.DataFrame(rows)
    va["video"] = pd.Categorical(
        va["video"],
        categories=[VIDEO_LABELS[v] for v in ALL_VIDEOS],
        ordered=True
    )
    return va


def plot_valence_arousal(df, outdir):
    print_section("VALENCE & AROUSAL SUMMARY STATISTICS")

    for v in ALL_VIDEOS:
        vc, ac = f"valence_{v}", f"arousal_{v}"
        print(f"{VIDEO_LABELS[v]:15s} "
              f"Valence: M={df[vc].mean():.2f}, SD={df[vc].std():.2f} \n\t\t "
              f"Arousal: M={df[ac].mean():.2f}, SD={df[ac].std():.2f}\n")

    va = build_valence_arousal_long(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, y, color, title in [
        (ax1, "Valence", "coral", "Valence Ratings by Video"),
        (ax2, "Arousal", "lightgreen", "Arousal Ratings by Video")
    ]:
        sns.boxplot(data=va, x="video", y=y, color=color, ax=ax, fliersize=0)
        sns.stripplot(data=va, x="video", y=y, color="black", alpha=0.4, size=3, jitter=True, ax=ax)

        ax.set(title=title, xlabel="", ylabel=y)
        ax.tick_params(axis='x', rotation=15)

    save_fig(fig, os.path.join(outdir, "Valence_Arousal_Boxplots.png"))

    # Circumplex
    circ = pd.DataFrame({
        "video": [VIDEO_LABELS[v] for v in ALL_VIDEOS],
        "valence": [df[f"valence_{v}"].mean() for v in ALL_VIDEOS],
        "arousal": [df[f"arousal_{v}"].mean() for v in ALL_VIDEOS],
    })

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(circ["valence"], circ["arousal"], s=80)

    for _, r in circ.iterrows():
        ax.annotate(r["video"], (r["valence"], r["arousal"]), xytext=(8, 6), textcoords="offset points")

    ax.axhline(5, linestyle='--', alpha=0.4)
    ax.axvline(5, linestyle='--', alpha=0.4)

    ax.set(xlim=(1, 9), ylim=(1, 9),
           xlabel="Valence", ylabel="Arousal",
           title="Circumplex Model")

    save_fig(fig, os.path.join(outdir, "Circumplex_Model.png"))


def build_headtracking_long(df):
    rows = [
        {
            "participant": r["participant"],
            "video": VIDEO_LABELS[v],
            "mean_speed": r[f"Mean Speed (Total)_v{v[1]}"],
            "sd_pitch": r[f"SD - Pitch (X)_v{v[1]}"],
            "sd_yaw": r[f"SD - Yaw (Y)_v{v[1]}"],
            "sd_roll": r[f"SD - Roll (Z)_v{v[1]}"],
            "total_movement": r[f"Total Movement Magnitude_v{v[1]}"],
        }
        for v in ALL_VIDEOS
        for _, r in df.iterrows()
    ]

    ht = pd.DataFrame(rows)
    ht["video"] = pd.Categorical(
        ht["video"],
        categories=[VIDEO_LABELS[v] for v in ALL_VIDEOS],
        ordered=True
    )
    return ht


def headtracking_stats(df, outdir):
    print_section("HEADTRACKING SUMMARY STATISTICS")

    ht = build_headtracking_long(df)

    fig, axes = plt.subplots(1, len(FIG4_CFG), figsize=(16, 4))

    for ax, (metric, title, ylabel, color) in zip(axes, FIG4_CFG):
        sns.boxplot(data=ht, x="video", y=metric, color=color, ax=ax, fliersize=0)
        sns.stripplot(data=ht, x="video", y=metric, color="black", alpha=0.4, size=3, jitter=True, ax=ax)

        ax.set(title=title, xlabel="", ylabel=ylabel)
        ax.tick_params(axis='x', rotation=15)

    save_fig(fig, os.path.join(outdir, "Headtracking_Boxplots.png"))


# =========================
# MAIN
# =========================
def main():
    df = pd.read_csv("output.csv")
    outdir = create_output_dir()

    print_header("DESCRIPTIVE STATISTICS")

    demographic_stats(df)
    clinical_stats(df)
    plot_histograms(df, outdir)
    plot_valence_arousal(df, outdir)
    headtracking_stats(df, outdir)


if __name__ == "__main__":
    main()