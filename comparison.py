# =============================================================================
# 0. IMPORTS & CONFIG
# =============================================================================
import os, warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import (
    shapiro,
    mannwhitneyu,
    ttest_ind,
    levene,
    spearmanr,
    pearsonr,
    kruskal,
    wilcoxon,
    friedmanchisquare,
)
from statsmodels.stats.multitest import multipletests
from statsmodels.formula.api import ols
import statsmodels.api as sm
from itertools import combinations

warnings.filterwarnings("ignore")

# ── User Config ───────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/ankita/BRSM/BRSM_Project1/360 Videos VR project/data")
HT_DIR = BASE_DIR / "headtracking-data"
DATA_FILE = BASE_DIR / "data.xlsx"
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

ALPHA = 0.05
MC_METHOD = "fdr_bh"
VIDEOS = {
    1: "Abandoned Buildings",
    2: "Beach",
    3: "Campus",
    4: "Horror (Nun)",
    5: "Tahiti Surf",
}

print("=" * 70)
print(" VR Head-Tracking Depression Study - Critical Replication")
print("=" * 70)

# =============================================================================
# 1. DATA LOADING & FEATURE EXTRACTION
# =============================================================================
print("\n[1] Loading data ...")
df_survey = pd.read_excel(DATA_FILE)
N = len(df_survey)
print(f"    N = {N} participants")


def extract_features(fpath: Path) -> dict:
    """
    Extract head-tracking features from a single CSV.
    Full-axis (X, Y, Z) extraction -- correcting FLAW 4.
    First row dropped (initialisation frame with zeros).
    """
    try:
        ht = pd.read_csv(fpath, on_bad_lines="skip").iloc[1:].reset_index(drop=True)
        feats = {}
        for ax in ["X", "Y", "Z"]:
            rcol = f"RotationChange{ax}"
            spcol = f"RotationSpeed{ax}"
            if rcol in ht.columns:
                vals = ht[rcol]
                feats[f"mean_rot_{ax}"] = vals.mean()
                feats[f"std_rot_{ax}"] = vals.std()
                feats[f"median_rot_{ax}"] = vals.median()
                feats[f"range_rot_{ax}"] = vals.max() - vals.min()
                feats[f"disp_rot_{ax}"] = vals.abs().sum()
            if spcol in ht.columns:
                sv = ht[spcol]
                feats[f"mean_speed_{ax}"] = sv.mean()
                feats[f"std_speed_{ax}"] = sv.std()
        if "RotationSpeedTotal" in ht.columns:
            feats["mean_speed_Total"] = ht["RotationSpeedTotal"].mean()
            feats["std_speed_Total"] = ht["RotationSpeedTotal"].std()
        return feats
    except Exception as e:
        print(f"      WARNING: {fpath}: {e}")
        return {}


records = []
for _, row in df_survey.iterrows():
    rec = {"participant": row["participant"]}
    for v in range(1, 6):
        fname = row.get(f"v{v}")
        if pd.notna(fname):
            fpath = HT_DIR / f"v{v}" / str(fname)
            for k, val in extract_features(fpath).items():
                rec[f"v{v}_{k}"] = val
    records.append(rec)

df_ht = pd.DataFrame(records)
df = df_survey.merge(df_ht, on="participant", how="left")
print(f"    Dataset shape after merging: {df.shape}")

# ---- Grouping (original paper style) ----------------------------------------
# Original paper: groups = {minimal+mild (PHQ 0-9)} vs {moderate+severe (PHQ >=10)}
df["phq_orig_group"] = np.where(df["score_phq"] >= 10, "mod_severe", "min_mild")
dep_orig = df[df["phq_orig_group"] == "mod_severe"]
nodep_orig = df[df["phq_orig_group"] == "min_mild"]

print(
    f"    Original grouping -- min/mild (PHQ<10): n={len(nodep_orig)}, "
    f"mod/severe (PHQ>=10): n={len(dep_orig)}"
)


# ---- Helper: BH correction --------------------------------------------------
def bh_correct(p_list):
    if len(p_list) == 0:
        return np.array([]), np.array([], dtype=bool)
    rej, pc, _, _ = multipletests(p_list, method=MC_METHOD)
    return pc, rej


# =============================================================================
# 2. REPLICATION: Original paper's analysis (FLAWED baseline)
#    - Tests only yaw (Y-axis) speed and SD
#    - Binary group split at PHQ >= 10
#    - No multiple comparison correction
#    - No anxiety covariate
# =============================================================================
print("\n" + "=" * 70)
print("[2] REPLICATION: Original paper's analysis (flawed)")
print("=" * 70)

orig_results = []
for v in range(1, 6):
    for feat in ["mean_speed_Y", "std_rot_Y"]:
        col = f"v{v}_{feat}"
        if col not in df.columns:
            continue
        g1 = dep_orig[col].dropna()
        g2 = nodep_orig[col].dropna()
        if len(g1) < 3 or len(g2) < 3:
            continue
        U, p = mannwhitneyu(g1, g2, alternative="two-sided")
        n1, n2 = len(g1), len(g2)
        rb = 1 - (2 * U) / (n1 * n2)
        orig_results.append(
            {
                "video": v,
                "feature": feat,
                "n_dep": n1,
                "n_nodep": n2,
                "mdn_dep": round(g1.median(), 3),
                "mdn_nodep": round(g2.median(), 3),
                "U": round(U, 2),
                "p_uncorrected": round(p, 4),
                "rank_biserial": round(rb, 3),
                "significant_UNCORRECTED": p < ALPHA,
            }
        )

df_orig = pd.DataFrame(orig_results)
print("\n  Replication results (NO multiple comparison correction):")
print(df_orig.to_string(index=False))
df_orig.to_csv(OUTPUT_DIR / "replication_original.csv", index=False)

n_orig_sig = df_orig["significant_UNCORRECTED"].sum()
print(
    f"\n  Original approach: {n_orig_sig}/{len(df_orig)} tests significant at alpha=.05 "
    f"(UNCORRECTED) -- expected ~{len(df_orig) * ALPHA:.1f} false positives by chance alone"
)

# =============================================================================
# 3. FLAW 1 FIX -- Continuous PHQ instead of dichotomization
#    "Median splits and extreme group analyses should be abandoned"
#    -- MacCallum et al., Psychological Methods (2002)
# =============================================================================
print("\n" + "=" * 70)
print("[3] FIX FLAW 1: Use PHQ-9 as continuous predictor (Spearman rho)")
print("    Rationale: Dichotomization discards variance, inflates Type-II error,")
print("    and manufactures artificial group thresholds.")
print("=" * 70)

cont_results = []
for v in range(1, 6):
    for feat in [
        "mean_speed_Y",
        "std_rot_Y",
        "mean_speed_X",
        "std_rot_X",
        "mean_speed_Z",
        "std_rot_Z",
        "mean_speed_Total",
    ]:
        col = f"v{v}_{feat}"
        if col not in df.columns:
            continue
        sub = df[["score_phq", col]].dropna()
        if len(sub) < 5:
            continue
        r, p = spearmanr(sub["score_phq"], sub[col])
        cont_results.append(
            {
                "video": v,
                "feature": feat,
                "n": len(sub),
                "rho": round(r, 3),
                "p": round(p, 4),
            }
        )

df_cont = pd.DataFrame(cont_results)
if len(df_cont):
    pc, rej = bh_correct(df_cont["p"].tolist())
    df_cont["p_corrected_BH"] = np.round(pc, 4)
    df_cont["significant"] = rej

    print("\n  Continuous PHQ ~ head-tracking (Spearman rho, BH-corrected):")
    print(df_cont.to_string(index=False))
    df_cont.to_csv(OUTPUT_DIR / "fix1_continuous_phq_corr.csv", index=False)

    sig_cont = df_cont[df_cont["significant"]]
    n_sig_cont = len(sig_cont)
    if n_sig_cont:
        print(f"\n  * {n_sig_cont} features survive correction:")
        print(
            sig_cont[["video", "feature", "rho", "p", "p_corrected_BH"]].to_string(
                index=False
            )
        )
    else:
        print("\n  No features survive BH correction with continuous PHQ.")
        print("  This suggests the original paper's group-level significance")
        print("  may not reflect a robust continuous dose-response relationship.")
else:
    n_sig_cont = 0

# =============================================================================
# 4. FLAW 2 FIX -- Multiple comparison correction
#    Re-run original analysis WITH BH-FDR correction applied
# =============================================================================
print("\n" + "=" * 70)
print("[4] FIX FLAW 2: Apply BH-FDR correction to original tests")
print("    Rationale: 10 uncorrected tests at alpha=.05 give ~0.5 expected false")
print("    positives. Standard practice requires familywise error control.")
print("=" * 70)

all_results_raw = []
for v in range(1, 6):
    for feat in [
        "mean_speed_Y",
        "std_rot_Y",
        "mean_speed_X",
        "std_rot_X",
        "mean_speed_Z",
        "std_rot_Z",
        "mean_speed_Total",
    ]:
        col = f"v{v}_{feat}"
        if col not in df.columns:
            continue
        g1 = dep_orig[col].dropna()
        g2 = nodep_orig[col].dropna()
        if len(g1) < 3 or len(g2) < 3:
            continue
        U, p = mannwhitneyu(g1, g2, alternative="two-sided")
        n1, n2 = len(g1), len(g2)
        rb = 1 - (2 * U) / (n1 * n2)
        all_results_raw.append(
            {
                "video": v,
                "feature": feat,
                "mdn_dep": round(g1.median(), 3),
                "mdn_nodep": round(g2.median(), 3),
                "n_dep": n1,
                "n_nodep": n2,
                "U": round(U, 2),
                "p_uncorrected": round(p, 4),
                "rank_biserial": round(rb, 3),
            }
        )

df_all = pd.DataFrame(all_results_raw)
pc, rej = bh_correct(df_all["p_uncorrected"].tolist())
df_all["p_corrected_BH"] = np.round(pc, 4)
df_all["sig_corrected"] = rej
df_all["sig_uncorrected"] = df_all["p_uncorrected"] < ALPHA

print(f"\n  Uncorrected significant: {df_all['sig_uncorrected'].sum()}/{len(df_all)}")
print(f"  BH-corrected significant: {df_all['sig_corrected'].sum()}/{len(df_all)}")
print("\n  Full results (sorted by uncorrected p):")
print(df_all.sort_values("p_uncorrected").to_string(index=False))
df_all.to_csv(OUTPUT_DIR / "fix2_mcc_corrected.csv", index=False)

lost = df_all[(df_all["sig_uncorrected"]) & (~df_all["sig_corrected"])]
if len(lost):
    print(f"\n  WARNING: {len(lost)} result(s) appear significant WITHOUT correction")
    print("    but are NOT significant AFTER BH correction (likely false positives):")
    print(
        lost[["video", "feature", "p_uncorrected", "p_corrected_BH"]].to_string(
            index=False
        )
    )

# =============================================================================
# 5. FLAW 3 FIX -- ANCOVA: partial out anxiety (GAD-7)
# =============================================================================
print("\n" + "=" * 70)
print("[5] FIX FLAW 3: ANCOVA controlling for GAD-7 anxiety")
print("    Rationale: PHQ and GAD are correlated. Without controlling for anxiety,")
print("    'depression effects' may be anxiety-driven hypervigilance or avoidance.")
print("=" * 70)

r_phq_gad, p_phq_gad = spearmanr(df["score_phq"], df["score_gad"])
print(f"\n  Sample: Spearman rho(PHQ, GAD) = {r_phq_gad:.3f}, p = {p_phq_gad:.4f}")
print(f"  Shared variance ~= {r_phq_gad**2 * 100:.1f}% -- covariance is substantial.")

df["dep_group_num"] = (df["score_phq"] >= 10).astype(int)

ancova_results = []
for v in range(1, 6):
    for feat in ["mean_speed_Y", "std_rot_Y", "mean_speed_Total"]:
        col = f"v{v}_{feat}"
        if col not in df.columns:
            continue
        sub = df[["dep_group_num", "score_gad", col]].dropna()
        sub = sub.rename(
            columns={col: "y", "dep_group_num": "group", "score_gad": "anxiety"}
        )
        if len(sub) < 10:
            continue
        try:
            model = ols("y ~ C(group) + anxiety", data=sub).fit()
            at = sm.stats.anova_lm(model, typ=2)
            if "C(group)" not in at.index:
                continue
            f_g = at.loc["C(group)", "F"]
            p_g = at.loc["C(group)", "PR(>F)"]
            ss_g = at.loc["C(group)", "sum_sq"]
            ss_res = at.loc["Residual", "sum_sq"]
            eta2p = ss_g / (ss_g + ss_res)
            ancova_results.append(
                {
                    "video": v,
                    "feature": feat,
                    "F": round(f_g, 3),
                    "p": round(p_g, 4),
                    "partial_eta2": round(eta2p, 4),
                }
            )
        except Exception as e:
            print(f"      ANCOVA error {col}: {e}")

df_ancova = pd.DataFrame(ancova_results)
n_ancova_sig = 0
if len(df_ancova):
    pc, rej = bh_correct(df_ancova["p"].tolist())
    df_ancova["p_corrected_BH"] = np.round(pc, 4)
    df_ancova["significant"] = rej
    n_ancova_sig = df_ancova["significant"].sum()
    print("\n  ANCOVA results (GAD-7 covariate, BH-corrected):")
    print(df_ancova.to_string(index=False))
    df_ancova.to_csv(OUTPUT_DIR / "fix3_ancova_anxiety.csv", index=False)

    if n_ancova_sig:
        print(
            f"\n  * {n_ancova_sig} features remain significant after anxiety control."
        )
        print("    These effects are genuinely depression-specific.")
    else:
        print("\n  No features survive anxiety control.")
        print(
            "  Original significant effects may be driven by anxiety, not depression."
        )

# =============================================================================
# 6. FLAW 4 FIX -- All-axis reporting (X, Y, Z)
# =============================================================================
print("\n" + "=" * 70)
print("[6] FIX FLAW 4: All-axis analysis (Pitch/X, Yaw/Y, Roll/Z)")
print("    Rationale: Reporting only Yaw without pre-registration is selective")
print("    outcome reporting. All axes must be tested and reported.")
print("=" * 70)

axis_results = []
for v in range(1, 6):
    for ax in ["X", "Y", "Z"]:
        for stat in ["mean_speed", "std_rot", "disp_rot"]:
            col = f"v{v}_{stat}_{ax}"
            if col not in df.columns:
                continue
            g1 = dep_orig[col].dropna()
            g2 = nodep_orig[col].dropna()
            if len(g1) < 3 or len(g2) < 3:
                continue
            U, p = mannwhitneyu(g1, g2, alternative="two-sided")
            n1, n2 = len(g1), len(g2)
            rb = 1 - (2 * U) / (n1 * n2)
            axis_results.append(
                {
                    "video": v,
                    "axis": ax,
                    "stat": stat,
                    "mdn_dep": round(g1.median(), 3),
                    "mdn_nodep": round(g2.median(), 3),
                    "U": round(U, 2),
                    "p_uncorrected": round(p, 4),
                    "rank_biserial": round(rb, 3),
                }
            )

df_axes = pd.DataFrame(axis_results)
pc, rej = bh_correct(df_axes["p_uncorrected"].tolist())
df_axes["p_corrected_BH"] = np.round(pc, 4)
df_axes["significant"] = rej
n_axes_sig = df_axes["significant"].sum()
print(
    f"\n  All-axis tests: {len(df_axes)} total, "
    f"{df_axes['p_uncorrected'].lt(ALPHA).sum()} uncorrected sig, "
    f"{n_axes_sig} BH-corrected sig"
)
sig_axes = df_axes[df_axes["significant"]]
if len(sig_axes):
    print("\n  Significant results across all axes:")
    print(
        sig_axes[
            [
                "video",
                "axis",
                "stat",
                "mdn_dep",
                "mdn_nodep",
                "p_uncorrected",
                "p_corrected_BH",
                "rank_biserial",
            ]
        ].to_string(index=False)
    )
else:
    print("  No features survive BH correction across all three axes.")
df_axes.to_csv(OUTPUT_DIR / "fix4_allaxis_results.csv", index=False)

# =============================================================================
# 7. FLAW 5 FIX -- Order-effect inspection
# =============================================================================
print("\n" + "=" * 70)
print("[7] FIX FLAW 5: Inspect order effects (fixed video presentation order)")
print("    Rationale: Without counterbalancing, position-in-sequence effects")
print("    confound video-content effects on head-tracking.")
print("=" * 70)

speed_trend = []
for v in range(1, 6):
    col = f"v{v}_mean_speed_Total"
    if col in df.columns:
        speed_trend.append(df[col].dropna().median())

videos_order = list(range(1, len(speed_trend) + 1))
r_trend, p_trend = (
    spearmanr(videos_order, speed_trend) if len(speed_trend) == 5 else (np.nan, np.nan)
)
if not np.isnan(r_trend):
    print(
        f"\n  Spearman rho(video position, median speed) = {r_trend:.3f}, p = {p_trend:.4f}"
    )
    print(f"  Median speeds per video (V1->V5): {[round(s, 2) for s in speed_trend]}")
    if p_trend < ALPHA:
        print("  WARNING: SIGNIFICANT monotonic trend detected across video position.")
        print("  Video-specific differences CANNOT be cleanly attributed to content.")
        print("  ORDER CONFOUND is present -- a randomized design is needed.")
    else:
        print("  Order-confound appears minimal (no significant trend).")
        print("  Caution warranted with N=40; power to detect subtle trends is low.")

std_trend = [
    df[f"v{v}_mean_speed_Total"].dropna().std()
    for v in range(1, 6)
    if f"v{v}_mean_speed_Total" in df.columns
]
r_std, p_std = (
    spearmanr(videos_order, std_trend) if len(std_trend) == 5 else (np.nan, np.nan)
)
if not np.isnan(r_std):
    print(f"  Spearman rho(video position, SD of speed) = {r_std:.3f}, p = {p_std:.4f}")

# =============================================================================
# 8. FLAW 6 FIX -- Post-hoc power analysis
# =============================================================================
print("\n" + "=" * 70)
print("[8] FIX FLAW 6: Post-hoc power analysis")
print("    Rationale: Without knowing power, significant results cannot be")
print("    trusted and non-significant results are uninformative.")
print("    Reference: Button et al. (2013) -- Nature Reviews Neuroscience.")
print("=" * 70)

n_dep_grp = len(dep_orig)
n_nodep_grp = len(nodep_orig)

try:
    from statsmodels.stats.power import tt_ind_solve_power

    ratio_val = n_nodep_grp / n_dep_grp
    print(f"\n  Group sizes: n_dep={n_dep_grp}, n_nodep={n_nodep_grp}")
    print(f"\n  Power to detect various effect sizes (two-tailed, alpha=.05):")
    print(f"  {'Effect (Cohen d)':<22} {'Power':>10}")
    for d in [0.2, 0.5, 0.8, 1.0]:
        pwr = tt_ind_solve_power(
            effect_size=d,
            nobs1=n_dep_grp,
            ratio=ratio_val,
            alpha=0.05,
            alternative="two-sided",
        )
        tag = (
            " <- small"
            if d == 0.2
            else " <- medium"
            if d == 0.5
            else " <- large"
            if d == 0.8
            else ""
        )
        print(f"  d = {d:<20} {pwr:>10.3f}{tag}")

    n_needed = tt_ind_solve_power(
        effect_size=0.5, power=0.8, alpha=0.05, alternative="two-sided"
    )
    print(f"\n  N needed per group for 80% power (d=0.5): {int(np.ceil(n_needed))}")
    print(f"  Current n_dep={n_dep_grp}: likely UNDERPOWERED for small-medium effects.")
    print("  Underpowered significant results are prone to effect size inflation")
    print("  (Type-M error; Gelman & Carlin, 2014).")
except ImportError:
    print("  statsmodels power analysis not available.")

# =============================================================================
# 9. FLAW 7 FIX -- Non-parametric tests for ordinal emotion ratings
# =============================================================================
print("\n" + "=" * 70)
print("[9] FIX FLAW 7: Kruskal-Wallis/Mann-Whitney for ordinal valence/arousal")
print("    Rationale: 9-point Likert-type ratings are ordinal. Parametric tests")
print("    (t-test, ANOVA) assume continuous, normally-distributed data.")
print("=" * 70)

emotion_results = []
for v in range(1, 6):
    for emo in ["valence", "arousal"]:
        col = f"{emo}_v{v}"
        if col not in df.columns:
            continue
        g1 = dep_orig[col].dropna()
        g2 = nodep_orig[col].dropna()
        if len(g1) < 3 or len(g2) < 3:
            continue
        U, p_mw = mannwhitneyu(g1, g2, alternative="two-sided")
        emotion_results.append(
            {
                "video": v,
                "measure": emo,
                "mdn_dep": round(g1.median(), 2),
                "mdn_nodep": round(g2.median(), 2),
                "U_MW": round(U, 2),
                "p_MW": round(p_mw, 4),
            }
        )

df_emo = pd.DataFrame(emotion_results)
if len(df_emo):
    pc, rej = bh_correct(df_emo["p_MW"].tolist())
    df_emo["p_corrected_BH"] = np.round(pc, 4)
    df_emo["significant"] = rej
    n_emo_sig = df_emo["significant"].sum()
    print("\n  Valence/Arousal between groups (Mann-Whitney, BH-corrected):")
    print(df_emo.to_string(index=False))
    df_emo.to_csv(OUTPUT_DIR / "fix7_emotion_nonparametric.csv", index=False)

    if n_emo_sig:
        print(
            f"\n  * Groups DO differ on {n_emo_sig} emotion ratings after correction."
        )
    else:
        print("\n  Non-parametric tests confirm: no significant group differences in")
        print(
            "  valence/arousal ratings. Original conclusion is upheld for this measure."
        )
else:
    n_emo_sig = 0

# =============================================================================
# 10. SUMMARY COMPARISON TABLE
# =============================================================================
print("\n" + "=" * 70)
print("[10] Summary: Original vs Corrected Conclusions")
print("=" * 70)

summary = [
    {
        "Flaw": "1 -- Dichotomization",
        "Original Paper": "Binary PHQ groups; loses continuous information",
        "Correction Applied": "Spearman rho with continuous PHQ",
        "Key Finding": f"{n_sig_cont} features survive BH with continuous PHQ",
    },
    {
        "Flaw": "2 -- No MCC",
        "Original Paper": f"{n_orig_sig}/10 tests significant (uncorrected)",
        "Correction Applied": "BH-FDR applied across all tests",
        "Key Finding": f"{df_all['sig_corrected'].sum()}/{len(df_all)} survive correction",
    },
    {
        "Flaw": "3 -- Anxiety confound",
        "Original Paper": "GAD-7 ignored; all effects attributed to depression",
        "Correction Applied": "ANCOVA with GAD-7 as covariate",
        "Key Finding": f"{n_ancova_sig} features survive anxiety control",
    },
    {
        "Flaw": "4 -- Yaw-only reporting",
        "Original Paper": "Only Y-axis reported (potential cherry-picking)",
        "Correction Applied": "X, Y, Z all tested and reported",
        "Key Finding": f"{n_axes_sig} features survive BH across all axes",
    },
    {
        "Flaw": "5 -- Order effects",
        "Original Paper": "Fixed order; not discussed or controlled",
        "Correction Applied": "Spearman trend test across video position",
        "Key Finding": f"rho={r_trend:.3f}, p={p_trend:.4f}",
    },
    {
        "Flaw": "6 -- No power analysis",
        "Original Paper": "Sample size not justified by power calculation",
        "Correction Applied": "Post-hoc power computed per effect size",
        "Key Finding": f"n_dep={n_dep_grp} -- likely underpowered for d<0.8",
    },
    {
        "Flaw": "7 -- Ordinal data mishandled",
        "Original Paper": "Parametric comparison implied for Likert ratings",
        "Correction Applied": "Mann-Whitney U for ordinal scale data",
        "Key Finding": f"{n_emo_sig}/10 emotion tests significant after correction",
    },
]

df_summary = pd.DataFrame(summary)
print(df_summary[["Flaw", "Key Finding"]].to_string(index=False))
df_summary.to_csv(OUTPUT_DIR / "summary_original_vs_corrected.csv", index=False)

# =============================================================================
# 11. EXPLORATORY EXTENSIONS
# =============================================================================
print("\n" + "=" * 70)
print("[11] Exploratory extensions")
print("=" * 70)

# 11a. PANAS mood change
print("\n  11a. PANAS mood change by PHQ group ...")
df["delta_PA"] = df["positive_affect_end"] - df["positive_affect_start"]
df["delta_NA"] = df["negative_affect_end"] - df["negative_affect_start"]
dep_orig = df[df["phq_orig_group"] == "mod_severe"]
nodep_orig = df[df["phq_orig_group"] == "min_mild"]
for label, col in [
    ("Delta Positive Affect", "delta_PA"),
    ("Delta Negative Affect", "delta_NA"),
]:
    g1 = dep_orig[col].dropna()
    g2 = nodep_orig[col].dropna()
    U, p = mannwhitneyu(g1, g2, alternative="two-sided")
    print(
        f"  {label}: Dep Mdn={g1.median():.2f}, NoDep Mdn={g2.median():.2f}, "
        f"U={U:.1f}, p={p:.4f}"
    )

# 11b. Presence x head-tracking
print("\n  11b. Presence (immersion) ~ head-tracking (Spearman rho, BH-corrected) ...")
pres_results = []
for v in range(1, 6):
    imm_col = f"immersion_v{v}"
    for feat in ["mean_speed_Y", "mean_speed_Total"]:
        col = f"v{v}_{feat}"
        if imm_col not in df.columns or col not in df.columns:
            continue
        sub = df[[imm_col, col]].dropna()
        if len(sub) < 5:
            continue
        r, p = spearmanr(sub[imm_col], sub[col])
        pres_results.append(
            {"video": v, "feature": feat, "rho": round(r, 3), "p": round(p, 4)}
        )

df_pres = pd.DataFrame(pres_results)
if len(df_pres):
    pc, rej = bh_correct(df_pres["p"].tolist())
    df_pres["p_corrected"] = np.round(pc, 4)
    df_pres["sig"] = rej
    print(df_pres.to_string(index=False))
    df_pres.to_csv(OUTPUT_DIR / "explore_presence_ht.csv", index=False)

# 11c. Video psychomotor profile (Friedman + Wilcoxon post-hoc)
print("\n  11c. Within-subjects video psychomotor comparison ...")
speed_cols = [
    f"v{v}_mean_speed_Total"
    for v in range(1, 6)
    if f"v{v}_mean_speed_Total" in df.columns
]
sw = df[speed_cols].dropna()
if len(sw) >= 5 and len(speed_cols) >= 2:
    stat_f, p_f = friedmanchisquare(*[sw[c] for c in speed_cols])
    print(f"  Friedman chi2({len(speed_cols) - 1}) = {stat_f:.3f}, p = {p_f:.4f}")
    ph = []
    for v1, v2 in combinations(range(1, 6), 2):
        c1, c2 = f"v{v1}_mean_speed_Total", f"v{v2}_mean_speed_Total"
        if c1 in df.columns and c2 in df.columns:
            sub2 = df[[c1, c2]].dropna()
            if len(sub2) < 3:
                continue
            W, p_w = wilcoxon(sub2[c1], sub2[c2])
            ph.append(
                {
                    "V1": VIDEOS[v1],
                    "V2": VIDEOS[v2],
                    "W": round(W, 2),
                    "p": round(p_w, 4),
                }
            )
    df_ph = pd.DataFrame(ph)
    if len(df_ph):
        pc, rej = bh_correct(df_ph["p"].tolist())
        df_ph["p_corrected"] = np.round(pc, 4)
        df_ph["sig"] = rej
        print(
            df_ph[["V1", "V2", "W", "p", "p_corrected", "sig"]].to_string(index=False)
        )
        df_ph.to_csv(OUTPUT_DIR / "explore_video_posthoc.csv", index=False)

# =============================================================================
# 12. VISUALISATIONS
# =============================================================================
print("\n[12] Generating figures ...")
sns.set_style("whitegrid")

# Fig 1: PHQ distribution with group boundary
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["score_phq"], bins=15, color="#5B8DB8", edgecolor="white", alpha=0.9)
ax.axvline(10, color="#C0392B", lw=2, ls="--", label="PHQ>=10 (mod/severe)")
ax.set_xlabel("PHQ-9 Total Score", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("PHQ-9 Distribution with Original Paper's Group Boundary", fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_phq_distribution.png", dpi=150)
plt.close()

# Fig 2: Uncorrected p vs BH-corrected p (shows how many flip)
if len(df_all):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#C0392B" if s else "#7F8C8D" for s in df_all["sig_uncorrected"]]
    ax.scatter(
        df_all["rank_biserial"],
        -np.log10(df_all["p_uncorrected"] + 1e-10),
        c=colors,
        s=70,
        alpha=0.8,
        zorder=3,
    )
    ax.axhline(-np.log10(ALPHA), color="#C0392B", ls="--", lw=1.5, label="alpha=.05")
    if df_all["sig_corrected"].any():
        max_sig_p = df_all[df_all["sig_corrected"]]["p_uncorrected"].max()
        ax.axhline(
            -np.log10(max_sig_p + 1e-10),
            color="#27AE60",
            ls="--",
            lw=1.5,
            label="BH threshold",
        )
    for _, row in df_all.iterrows():
        ax.annotate(
            f"V{int(row['video'])} {row['feature'][:8]}",
            (row["rank_biserial"], -np.log10(row["p_uncorrected"] + 1e-10)),
            fontsize=6,
            alpha=0.7,
            xytext=(4, 2),
            textcoords="offset points",
        )
    ax.set_xlabel("Rank-Biserial Correlation (Effect Size)", fontsize=11)
    ax.set_ylabel("-log10(p) Uncorrected", fontsize=11)
    ax.set_title(
        "Uncorrected vs BH-Corrected Significance\nRed = 'significant' without correction",
        fontsize=12,
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_pvalue_correction.png", dpi=150)
    plt.close()

# Fig 3: Continuous PHQ vs Yaw speed scatter plots per video
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for i, v in enumerate(range(1, 6)):
    col = f"v{v}_mean_speed_Y"
    if col not in df.columns:
        continue
    sub = df[["score_phq", col]].dropna()
    r, p = spearmanr(sub["score_phq"], sub[col])
    axes[i].scatter(sub["score_phq"], sub[col], alpha=0.6, s=40, color="#2E86AB")
    m, b = np.polyfit(sub["score_phq"], sub[col], 1)
    x_line = np.linspace(sub["score_phq"].min(), sub["score_phq"].max(), 50)
    axes[i].plot(x_line, m * x_line + b, color="#E74C3C", lw=1.5)
    axes[i].set_title(f"V{v}: {VIDEOS[v][:10]}\nrho={r:.2f}, p={p:.3f}", fontsize=8)
    axes[i].set_xlabel("PHQ-9")
    axes[i].set_ylabel("Mean Yaw Speed" if i == 0 else "")
plt.suptitle(
    "Fix Flaw 1: Continuous PHQ-9 vs Yaw Speed (vs. artificial binary split)", y=1.02
)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_continuous_phq_vs_yaw.png", dpi=150)
plt.close()

# Fig 4: All-axis p-value heatmap
ax_feat_pairs = [
    ("X", "mean_speed"),
    ("Y", "mean_speed"),
    ("Z", "mean_speed"),
    ("X", "std_rot"),
    ("Y", "std_rot"),
    ("Z", "std_rot"),
]
pval_matrix = np.ones((5, len(ax_feat_pairs)))
for j, (ax_, stat) in enumerate(ax_feat_pairs):
    for v in range(1, 6):
        col = f"v{v}_{stat}_{ax_}"
        if col not in df.columns:
            continue
        g1 = dep_orig[col].dropna()
        g2 = nodep_orig[col].dropna()
        if len(g1) < 3 or len(g2) < 3:
            continue
        _, p = mannwhitneyu(g1, g2, alternative="two-sided")
        pval_matrix[v - 1, j] = p

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(
    -np.log10(pval_matrix + 1e-10), cmap="YlOrRd", aspect="auto", vmin=0, vmax=3
)
plt.colorbar(im, ax=ax, label="-log10(p)")
ax.set_xticks(range(len(ax_feat_pairs)))
ax.set_xticklabels([f"{s}_{a}" for a, s in ax_feat_pairs], rotation=30, ha="right")
ax.set_yticks(range(5))
ax.set_yticklabels([f"V{v}: {VIDEOS[v][:12]}" for v in range(1, 6)])
ax.set_title(
    "Fix Flaw 4: All-Axis p-values\n(warmer = more significant; UNCORRECTED)",
    fontsize=12,
)
for i in range(5):
    for j in range(len(ax_feat_pairs)):
        val = pval_matrix[i, j]
        ax.text(
            j,
            i,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color="black" if val > 0.05 else "white",
        )
plt.tight_layout()
plt.savefig(FIG_DIR / "fig4_allaxis_heatmap.png", dpi=150)
plt.close()

# Fig 5: Power curve
try:
    from statsmodels.stats.power import tt_ind_solve_power

    d_range = np.linspace(0.1, 1.5, 100)
    ratio_val = n_nodep_grp / n_dep_grp
    pwr_curve = [
        tt_ind_solve_power(
            effect_size=d,
            nobs1=n_dep_grp,
            ratio=ratio_val,
            alpha=0.05,
            alternative="two-sided",
        )
        for d in d_range
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(d_range, pwr_curve, color="#2E86AB", lw=2.5, label=f"n_dep={n_dep_grp}")
    ax.axhline(0.8, color="#C0392B", ls="--", lw=1.5, label="80% power threshold")
    ax.axvline(0.5, color="#E67E22", ls=":", lw=1.5, label="Medium effect d=0.5")
    ax.fill_between(
        d_range,
        pwr_curve,
        0.8,
        where=[p < 0.8 for p in pwr_curve],
        alpha=0.15,
        color="red",
        label="Underpowered zone",
    )
    ax.set_xlabel("Cohen's d")
    ax.set_ylabel("Statistical Power")
    ax.set_title(
        f"Fix Flaw 6: Power Curve (n_dep={n_dep_grp}, alpha=.05)\n"
        "Study is underpowered for small-medium effects",
        fontsize=11,
    )
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_power_curve.png", dpi=150)
    plt.close()
except Exception as e:
    print(f"  Power plot skipped: {e}")

# Fig 6: ANCOVA F-stats
if len(df_ancova):
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [f"V{r['video']} {r['feature']}" for _, r in df_ancova.iterrows()]
    f_vals = df_ancova["F"].values
    colors = ["#27AE60" if s else "#E74C3C" for s in df_ancova["significant"]]
    ax.barh(range(len(labels)), f_vals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("F-statistic (ANCOVA group effect, controlling GAD-7)")
    ax.set_title(
        "Fix Flaw 3: Depression Group Effect After Partialling Out Anxiety\n"
        "Green = BH-significant; Red = not significant",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_ancova_fstats.png", dpi=150)
    plt.close()

print(f"\n  All figures saved to: {FIG_DIR}")
print("\n" + "=" * 70)
print("Analysis complete. All results in:", OUTPUT_DIR)
print("=" * 70)
