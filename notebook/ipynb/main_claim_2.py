import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, t

# ============================================================
# Claim 2: Scalarized fairness--utility trade-off analysis
#
# Goal:
#   Test whether POST, defined as ROC/EqOdds, has a better
#   EOD/SPD-oriented fairness--utility trade-off than PRE or IN.
#
# New expected folder structure:
#
# root/
#   Adult/
#     output/<ML_MODEL>/Adult/<DP_SYNTHESIZER>/results/*.csv
#
#   ACSIncome/
#     output/<ML_MODEL>/ACSIncome/<DP_SYNTHESIZER>/results/*.csv
#
#   Compas/
#     output/<ML_MODEL>/Compas/<DP_SYNTHESIZER>/results/*.csv
#
#   BoD/
#     BoD-5/
#       output/<ML_MODEL>/BoD-5/<DP_SYNTHESIZER>/results/*.csv
#
# Score:
#   S_a = sqrt(w_U * (1 - U_a)^2 + w_F * |m_a|^2)
#
# Lower S_a means better trade-off:
#   - U_a should be high
#   - |m_a| should be low
#
# Stage score:
#   S_stage = min_{method in stage} S_method
#
# Wilcoxon test:
#   d_i = S_POST_i - S_OTHER_i
#
# H0: median(d_i) = 0
# H1: median(d_i) < 0
#
# Negative d_i means POST has a better scalarized trade-off.
# ============================================================


# =========================
# Configuration
# =========================

DP_SYNTHESIZER = "aim"
ML_MODELS = ["XGB", "RF", "LR"]

DATASET_CONFIGS = [
    {
        "dataset": "adult",
        "display_name": "Adult",
        "path_dataset": "Adult",
        "bod_combo": None,
    },
    {
        "dataset": "folktables",
        "display_name": "ACSIncome",
        "path_dataset": "ACSIncome",
        "bod_combo": None,
    },
    {
        "dataset": "compas",
        "display_name": "COMPAS",
        "path_dataset": "Compas",
        "bod_combo": None,
    },
    {
        "dataset": "bod",
        "display_name": "BoD-Config-5",
        "path_dataset": "BoD-5",
        "bod_combo": 5,
    },
]

UTILITY_METRICS = ["ACC", "F1"]

# Main Claim 2 condition: EOD/SPD-oriented trade-offs.
FAIRNESS_METRICS = ["EOD", "SPD"]

PRE_METHODS = ["dir", "lfr", "reweigh"]
IN_METHODS = ["egr", "gsr"]

# Claim 2 focuses on the post-processing methods emphasized in the paper.
# CEOP is intentionally excluded because its behavior is more heterogeneous.
POST_METHODS = ["roc", "eqodds"]

STAGE_METHODS = {
    "PRE": PRE_METHODS,
    "IN": IN_METHODS,
    "POST": POST_METHODS,
}

METHOD_STAGE = {
    method: stage
    for stage, methods in STAGE_METHODS.items()
    for method in methods
}

ALL_METHODS = PRE_METHODS + IN_METHODS + POST_METHODS

ALPHA = 0.05

# Equal weights by default.
# Since both utility and |fairness| are assumed to be in [0,1],
# this is a distance to the ideal point (U=1, |m|=0).
UTILITY_WEIGHT = 0.5
FAIRNESS_WEIGHT = 0.5

# If True, utility and fairness are min-max normalized within each
# dataset/model/utility/fairness block before scoring.
# If False, use raw U and |m| directly. Recommended here: False.
USE_MINMAX_NORMALIZATION = False

OUTPUT_DETAILED_SCORES = (
    f"claim2_scalarized_tradeoff_scores_all_models_{DP_SYNTHESIZER}.csv"
)

OUTPUT_STAGE_SCORES = (
    f"claim2_scalarized_best_stage_scores_all_models_{DP_SYNTHESIZER}.csv"
)

OUTPUT_DELTAS = (
    f"claim2_scalarized_post_vs_pre_in_deltas_all_models_{DP_SYNTHESIZER}.csv"
)

OUTPUT_GLOBAL_RESULTS = (
    f"claim2_scalarized_wilcoxon_global_all_models_{DP_SYNTHESIZER}.csv"
)

OUTPUT_MODEL_BREAKDOWN = (
    f"claim2_scalarized_wilcoxon_by_model_{DP_SYNTHESIZER}.csv"
)

OUTPUT_LATEX_TABLE = (
    f"claim2_scalarized_wilcoxon_paper_table_all_models_{DP_SYNTHESIZER}.tex"
)


# ============================================================
# Path helpers
# ============================================================

def find_case_insensitive_dir(parent_dir, candidates):
    """
    Find a directory inside parent_dir matching one candidate, case-insensitively.
    """
    if not os.path.isdir(parent_dir):
        return None

    children = {
        child.lower(): child
        for child in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, child))
    }

    for candidate in candidates:
        key = str(candidate).lower()

        if key in children:
            return os.path.join(parent_dir, children[key])

    return None


def resolve_results_root():
    """
    Resolve the root directory containing:
        Adult/
        ACSIncome/
        Compas/
        BoD/

    If automatic detection fails, define RESULTS_BASE_DIR before running.
    """
    top_level_dataset_names = ["Adult", "ACSIncome", "Compas", "BoD"]

    candidates = []

    if "RESULTS_BASE_DIR" in globals():
        user_path = os.path.abspath(os.path.expanduser(str(RESULTS_BASE_DIR)))
        candidates.append(user_path)

        base = os.path.basename(user_path).lower()

        if base in [name.lower() for name in top_level_dataset_names]:
            candidates.append(os.path.dirname(user_path))

        if base.startswith("bod-"):
            parent = os.path.dirname(user_path)

            if os.path.basename(parent).lower() == "bod":
                candidates.append(os.path.dirname(parent))

    cwd = os.path.abspath(os.getcwd())
    candidates.append(cwd)

    parent = cwd
    while True:
        candidates.append(parent)

        new_parent = os.path.dirname(parent)

        if new_parent == parent:
            break

        parent = new_parent

    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue

        for dataset_name in top_level_dataset_names:
            if os.path.isdir(os.path.join(candidate, dataset_name)):
                return os.path.abspath(candidate)

    raise FileNotFoundError(
        "Could not find the root directory containing Adult, ACSIncome, Compas, or BoD.\n\n"
        f"Current working directory:\n{cwd}\n\n"
        "Expected structure:\n"
        "root/Adult/output/<ML_MODEL>/Adult/<DP_SYNTHESIZER>/results/\n"
        "root/ACSIncome/output/<ML_MODEL>/ACSIncome/<DP_SYNTHESIZER>/results/\n"
        "root/Compas/output/<ML_MODEL>/Compas/<DP_SYNTHESIZER>/results/\n"
        "root/BoD/BoD-5/output/<ML_MODEL>/BoD-5/<DP_SYNTHESIZER>/results/\n\n"
        "Define RESULTS_BASE_DIR manually, for example:\n"
        "RESULTS_BASE_DIR = r'C:\\path\\to\\DP-Benchmark'"
    )


def dataset_dir_candidates(path_dataset):
    """
    Candidate names for regular dataset folders.
    """
    if path_dataset == "Adult":
        return ["Adult", "ADULT", "adult"]

    if path_dataset == "ACSIncome":
        return [
            "ACSIncome",
            "ACSincome",
            "acsincome",
            "ACS_Income",
            "acs_income",
            "folktables",
            "Folktables",
        ]

    if path_dataset == "Compas":
        return ["Compas", "COMPAS", "compas"]

    return [path_dataset, path_dataset.lower(), path_dataset.upper()]


def bod_combo_candidates(bod_combo):
    """
    Candidate names for BoD combo folders.
    """
    return [
        f"BoD-{bod_combo}",
        f"BOD-{bod_combo}",
        f"bod-{bod_combo}",
        f"BoD_{bod_combo}",
        f"BOD_{bod_combo}",
        f"bod_{bod_combo}",
    ]


def get_results_dir(results_root, config, ml_model, dp_synthesizer):
    """
    Resolve exact benchmark results directory for one dataset/model/synth.
    """
    path_dataset = config["path_dataset"]
    bod_combo = config["bod_combo"]

    model_candidates = [
        str(ml_model),
        str(ml_model).lower(),
        str(ml_model).upper(),
    ]

    synth_candidates = [
        str(dp_synthesizer),
        str(dp_synthesizer).lower(),
        str(dp_synthesizer).upper(),
    ]

    if bod_combo is not None:
        # root/BoD/BoD-5/output/<ML_MODEL>/BoD-5/<SYNTH>/results/
        bod_root = find_case_insensitive_dir(
            results_root,
            ["BoD", "BOD", "bod"],
        )

        if bod_root is None:
            return None

        bod_combo_dir = find_case_insensitive_dir(
            bod_root,
            bod_combo_candidates(bod_combo),
        )

        if bod_combo_dir is None:
            return None

        output_dir = find_case_insensitive_dir(
            bod_combo_dir,
            ["output", "Output"],
        )

        if output_dir is None:
            return None

        model_dir = find_case_insensitive_dir(
            output_dir,
            model_candidates,
        )

        if model_dir is None:
            return None

        dataset_dir_2 = find_case_insensitive_dir(
            model_dir,
            bod_combo_candidates(bod_combo),
        )

        if dataset_dir_2 is None:
            return None

        synth_dir = find_case_insensitive_dir(
            dataset_dir_2,
            synth_candidates,
        )

        if synth_dir is None:
            return None

        return find_case_insensitive_dir(
            synth_dir,
            ["results", "Results"],
        )

    # root/<DATASET>/output/<ML_MODEL>/<DATASET>/<SYNTH>/results/
    dataset_dir_1 = find_case_insensitive_dir(
        results_root,
        dataset_dir_candidates(path_dataset),
    )

    if dataset_dir_1 is None:
        return None

    output_dir = find_case_insensitive_dir(
        dataset_dir_1,
        ["output", "Output"],
    )

    if output_dir is None:
        return None

    model_dir = find_case_insensitive_dir(
        output_dir,
        model_candidates,
    )

    if model_dir is None:
        return None

    dataset_dir_2 = find_case_insensitive_dir(
        model_dir,
        dataset_dir_candidates(path_dataset),
    )

    if dataset_dir_2 is None:
        return None

    synth_dir = find_case_insensitive_dir(
        dataset_dir_2,
        synth_candidates,
    )

    if synth_dir is None:
        return None

    return find_case_insensitive_dir(
        synth_dir,
        ["results", "Results"],
    )


def is_benchmark_result_csv(file_name, dp_synthesizer):
    """
    Keep benchmark result CSV files and exclude log CSV files.
    """
    file_lower = str(file_name).lower()
    synth_lower = str(dp_synthesizer).lower()

    if not file_lower.endswith(".csv"):
        return False

    if "-log.csv" in file_lower:
        return False

    if not file_lower.startswith("benchmark_results_seeds_"):
        return False

    if f"_synth_{synth_lower}" not in file_lower:
        return False

    return True


def extract_seed_from_filename(file_path):
    """
    Extract seed from:
        benchmark_results_seeds_5_eps_..._synth_aim.csv
    """
    base = os.path.basename(str(file_path))

    token_start = "benchmark_results_seeds_"
    token_end = "_eps_"

    if token_start not in base or token_end not in base:
        return np.nan

    try:
        seed_part = base.split(token_start, 1)[1].split(token_end, 1)[0]

        if "_" in seed_part:
            return seed_part

        return int(seed_part)

    except Exception:
        return np.nan


def read_dataset_model_csvs(results_root, config, ml_model, dp_synthesizer):
    """
    Read all seed-level benchmark CSVs for one dataset/model/synth.
    """
    results_dir = get_results_dir(
        results_root=results_root,
        config=config,
        ml_model=ml_model,
        dp_synthesizer=dp_synthesizer,
    )

    if results_dir is None or not os.path.isdir(results_dir):
        print(
            f"[SKIP] Results directory not found for "
            f"ML={ml_model}, Dataset={config['display_name']}, "
            f"Synth={dp_synthesizer}, searched_dir={results_dir}"
        )
        return None

    csv_files = []

    for file_name in os.listdir(results_dir):
        if not is_benchmark_result_csv(file_name, dp_synthesizer):
            continue

        file_path = os.path.join(results_dir, file_name)

        if os.path.isfile(file_path):
            csv_files.append(file_path)

    csv_files = sorted(csv_files, key=lambda p: os.path.basename(p))

    if len(csv_files) == 0:
        print(
            f"[SKIP] No benchmark CSV files found for "
            f"ML={ml_model}, Dataset={config['display_name']}, "
            f"Synth={dp_synthesizer}, results_dir={results_dir}"
        )
        return None

    dfs = []

    for csv_file in csv_files:
        tmp = pd.read_csv(csv_file)
        tmp = tmp.replace([np.inf, -np.inf], np.nan)

        if "Seed" not in tmp.columns:
            tmp["Seed"] = extract_seed_from_filename(csv_file)

        tmp["Source-File"] = os.path.basename(csv_file)
        tmp["Source-Path"] = csv_file

        dfs.append(tmp)

    df = pd.concat(dfs, ignore_index=True)

    print(
        f"[READ] ML={ml_model}, Dataset={config['display_name']}, "
        f"Synth={dp_synthesizer}: {len(csv_files)} CSV files, {len(df)} rows"
    )
    print(f"[DIR] {results_dir}")

    return df


# ============================================================
# Helper functions
# ============================================================

def detect_seed_column(df):
    possible_seed_columns = [
        "Seed", "seed", "RandomSeed", "random_seed", "run", "Run"
    ]

    for col in possible_seed_columns:
        if col in df.columns:
            return col

    raise ValueError(
        "No seed column found. Expected one of: "
        f"{possible_seed_columns}"
    )


def ensure_f1(df):
    if "F1" in df.columns:
        return df

    if {"PREC", "REC"}.issubset(df.columns):
        df = df.copy()
        df["F1"] = np.where(
            (df["PREC"] + df["REC"]) > 0,
            2 * df["PREC"] * df["REC"] / (df["PREC"] + df["REC"]),
            0.0,
        )
        return df

    raise ValueError("F1 requires either an F1 column or PREC and REC columns.")


def minmax_normalize(values):
    values = np.asarray(values, dtype=float)

    if np.all(np.isnan(values)):
        return values

    v_min = np.nanmin(values)
    v_max = np.nanmax(values)

    if np.isclose(v_max, v_min):
        return np.zeros_like(values)

    return (values - v_min) / (v_max - v_min)


def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    n = len(values)

    if n == 0:
        return np.nan, np.nan, np.nan, 0

    mean = float(np.mean(values))

    if n == 1:
        return mean, mean, mean, 1

    sem = np.std(values, ddof=1) / np.sqrt(n)
    half_width = t.ppf(0.975, df=n - 1) * sem

    return mean, float(mean - half_width), float(mean + half_width), n


def run_wilcoxon_less(deltas):
    """
    One-sided Wilcoxon signed-rank test.

    d_i = S_POST_i - S_OTHER_i

    H0: median(d_i) = 0
    H1: median(d_i) < 0

    Negative values indicate that POST has a better scalarized trade-off.
    """

    d = np.asarray(deltas, dtype=float)
    d = d[~np.isnan(d)]

    if len(d) < 2:
        return None

    mean_delta, ci_low, ci_high, n = mean_ci95(d)

    median_delta = float(np.median(d))
    win_rate = float(np.mean(d < 0))
    loss_rate = float(np.mean(d > 0))
    tie_rate = float(np.mean(d == 0))

    if np.allclose(d, 0):
        return {
            "n_pairs": n,
            "mean_delta": mean_delta,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "median_delta": median_delta,
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "tie_rate": tie_rate,
            "wilcoxon_stat": np.nan,
            "p_value": 1.0,
            "significant": False,
            "post_better": False,
        }

    stat, p_value = wilcoxon(
        d,
        alternative="less",
        zero_method="wilcox",
        mode="auto",
    )

    return {
        "n_pairs": n,
        "mean_delta": mean_delta,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "median_delta": median_delta,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "tie_rate": tie_rate,
        "wilcoxon_stat": float(stat),
        "p_value": float(p_value),
        "significant": bool((p_value < ALPHA) and (mean_delta < 0)),
        "post_better": bool(mean_delta < 0),
    }


def format_p_value(x):
    if pd.isna(x):
        return "NA"
    if x == 0 or x < 1e-300:
        return r"$<10^{-300}$"
    return f"{x:.2e}"


def compute_scalarized_tradeoff_scores(
    df,
    seed_col,
    ml_model,
    dataset_key,
    dataset_display,
    utility_metric,
    fairness_metric,
):
    """
    Compute scalarized fairness--utility score for all selected methods.

    S_a = sqrt(w_U * (1 - U_a)^2 + w_F * |m_a|^2)

    Lower score is better.
    """

    required_columns = [
        seed_col,
        "Epsilon",
        "Fair-Method",
        utility_metric,
        fairness_metric,
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for scalarized analysis: {missing}")

    fair_df = df[
        df["Fair-Method"].isin(ALL_METHODS)
        & df["Epsilon"].notna()
    ][required_columns].copy()

    fair_df = fair_df.dropna(subset=[utility_metric, fairness_metric])

    if fair_df.empty:
        return pd.DataFrame()

    # If duplicates exist for the same seed/epsilon/method, average them.
    fair_df = (
        fair_df
        .groupby([seed_col, "Epsilon", "Fair-Method"], dropna=False)
        .agg({
            utility_metric: "mean",
            fairness_metric: "mean",
        })
        .reset_index()
    )

    fair_df["ml_model"] = ml_model
    fair_df["dataset"] = dataset_key
    fair_df["dataset_display"] = dataset_display
    fair_df["utility_metric"] = utility_metric
    fair_df["fairness_metric"] = fairness_metric
    fair_df["stage"] = fair_df["Fair-Method"].map(METHOD_STAGE)

    fair_df = fair_df.rename(columns={seed_col: "seed"})

    fair_df["utility_raw"] = fair_df[utility_metric].astype(float)
    fair_df["abs_fairness_raw"] = np.abs(fair_df[fairness_metric].astype(float))

    if USE_MINMAX_NORMALIZATION:
        fair_df["utility_score_component"] = minmax_normalize(fair_df["utility_raw"])
        fair_df["fairness_score_component"] = minmax_normalize(fair_df["abs_fairness_raw"])
    else:
        fair_df["utility_score_component"] = fair_df["utility_raw"]
        fair_df["fairness_score_component"] = fair_df["abs_fairness_raw"]

    fair_df["tradeoff_score"] = np.sqrt(
        UTILITY_WEIGHT * (1.0 - fair_df["utility_score_component"]) ** 2
        + FAIRNESS_WEIGHT * fair_df["fairness_score_component"] ** 2
    )

    return fair_df


def get_best_stage_scores(score_df):
    """
    For each model/dataset/seed/epsilon/utility/fairness, select the
    best scalarized score inside each stage:

    S_stage = min_{method in stage} S_method
    """

    group_cols = [
        "ml_model",
        "dataset",
        "dataset_display",
        "seed",
        "Epsilon",
        "utility_metric",
        "fairness_metric",
        "stage",
    ]

    best = (
        score_df
        .groupby(group_cols, dropna=False)["tradeoff_score"]
        .min()
        .reset_index()
    )

    index_cols = [
        "ml_model",
        "dataset",
        "dataset_display",
        "seed",
        "Epsilon",
        "utility_metric",
        "fairness_metric",
    ]

    pivot = best.pivot_table(
        index=index_cols,
        columns="stage",
        values="tradeoff_score",
        aggfunc="first",
    ).reset_index()

    pivot.columns.name = None

    return pivot


def build_stage_deltas(stage_scores_df):
    """
    Build paired stage differences.

    d_i = S_POST - S_OTHER

    Negative d_i means POST has better scalarized trade-off.
    """

    delta_frames = []

    for comparison, other_stage in [
        ("POST_vs_PRE", "PRE"),
        ("POST_vs_IN", "IN"),
    ]:
        if "POST" not in stage_scores_df.columns or other_stage not in stage_scores_df.columns:
            continue

        sub = stage_scores_df.dropna(subset=["POST", other_stage]).copy()
        sub["comparison"] = comparison
        sub["other_stage"] = other_stage
        sub["delta"] = sub["POST"] - sub[other_stage]

        delta_frames.append(sub)

    if len(delta_frames) == 0:
        return pd.DataFrame()

    return pd.concat(delta_frames, ignore_index=True)


if __name__ == "__main__":

    # ============================================================
    # Step 1: Read original CSV files and compute scalarized scores
    # ============================================================

    RESULTS_ROOT = resolve_results_root()

    print(f"[RESULTS_ROOT] {RESULTS_ROOT}")
    print(f"[DP_SYNTHESIZER] {DP_SYNTHESIZER}")

    all_score_frames = []

    for ml_model in ML_MODELS:
        print("\n" + "=" * 80)
        print(f"Reading files and computing scalarized scores for ML model: {ml_model}")
        print("=" * 80)

        for config in DATASET_CONFIGS:
            dataset_key = config["dataset"]
            dataset_display = config["display_name"]

            df = read_dataset_model_csvs(
                results_root=RESULTS_ROOT,
                config=config,
                ml_model=ml_model,
                dp_synthesizer=DP_SYNTHESIZER,
            )

            if df is None:
                continue

            df = df.replace([np.inf, -np.inf], np.nan)

            seed_col = detect_seed_column(df)
            df = ensure_f1(df)

            unique_seeds = sorted(df[seed_col].dropna().unique())
            unique_eps = sorted(df["Epsilon"].dropna().unique())

            print(
                f"       ML={ml_model}, Dataset={dataset_display}, "
                f"n_seeds={len(unique_seeds)}, epsilons={unique_eps}"
            )

            for utility_metric in UTILITY_METRICS:
                for fairness_metric in FAIRNESS_METRICS:
                    score_df = compute_scalarized_tradeoff_scores(
                        df=df,
                        seed_col=seed_col,
                        ml_model=ml_model,
                        dataset_key=dataset_key,
                        dataset_display=dataset_display,
                        utility_metric=utility_metric,
                        fairness_metric=fairness_metric,
                    )

                    if score_df.empty:
                        print(
                            f"[SKIP] Empty score data for "
                            f"{ml_model}, {dataset_display}, "
                            f"{utility_metric}, {fairness_metric}"
                        )
                        continue

                    all_score_frames.append(score_df)

    if len(all_score_frames) == 0:
        raise ValueError("No scalarized score data were generated.")

    scores_df = pd.concat(all_score_frames, ignore_index=True)
    scores_df.to_csv(OUTPUT_DETAILED_SCORES, index=False)

    print(f"\n[SAVE] Detailed scalarized scores: {OUTPUT_DETAILED_SCORES}")


    # ============================================================
    # Step 2: Select best score per stage and build paired deltas
    # ============================================================

    stage_scores_df = get_best_stage_scores(scores_df)
    stage_scores_df.to_csv(OUTPUT_STAGE_SCORES, index=False)

    print(f"[SAVE] Best stage scores: {OUTPUT_STAGE_SCORES}")

    deltas_df = build_stage_deltas(stage_scores_df)
    deltas_df.to_csv(OUTPUT_DELTAS, index=False)

    print(f"[SAVE] POST vs PRE/IN deltas: {OUTPUT_DELTAS}")


    # ============================================================
    # Step 3: Global Wilcoxon tests across all models and datasets
    # ============================================================

    global_results = []

    for comparison in ["POST_vs_PRE", "POST_vs_IN"]:
        sub = deltas_df[deltas_df["comparison"] == comparison].copy()

        if sub.empty:
            continue

        test_result = run_wilcoxon_less(sub["delta"].to_numpy())

        if test_result is None:
            continue

        global_results.append({
            "comparison": comparison,
            "dp_synthesizer": DP_SYNTHESIZER,
            "utility_metrics": ",".join(UTILITY_METRICS),
            "fairness_metrics": ",".join(FAIRNESS_METRICS),
            "n_unique_models": sub["ml_model"].nunique(),
            "n_unique_datasets": sub["dataset_display"].nunique(),
            "n_unique_model_dataset_seed_triples": (
                sub[["ml_model", "dataset_display", "seed"]]
                .drop_duplicates()
                .shape[0]
            ),
            "n_unique_epsilons": sub["Epsilon"].nunique(),
            "utility_weight": UTILITY_WEIGHT,
            "fairness_weight": FAIRNESS_WEIGHT,
            "use_minmax_normalization": USE_MINMAX_NORMALIZATION,
            **test_result,
        })

    global_results_df = pd.DataFrame(global_results)
    global_results_df.to_csv(OUTPUT_GLOBAL_RESULTS, index=False)

    print("\n=== Global scalarized Claim 2 results ===")
    print(global_results_df)
    print(f"\n[SAVE] Global results: {OUTPUT_GLOBAL_RESULTS}")


    # ============================================================
    # Step 4: Breakdown by ML model
    # ============================================================

    model_results = []

    for ml_model in sorted(deltas_df["ml_model"].unique()):
        model_df = deltas_df[deltas_df["ml_model"] == ml_model].copy()

        for comparison in ["POST_vs_PRE", "POST_vs_IN"]:
            sub = model_df[model_df["comparison"] == comparison].copy()

            if sub.empty:
                continue

            test_result = run_wilcoxon_less(sub["delta"].to_numpy())

            if test_result is None:
                continue

            model_results.append({
                "ml_model": ml_model,
                "comparison": comparison,
                "n_unique_datasets": sub["dataset_display"].nunique(),
                "n_unique_seed_dataset_pairs": (
                    sub[["dataset_display", "seed"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "n_unique_epsilons": sub["Epsilon"].nunique(),
                "utility_weight": UTILITY_WEIGHT,
                "fairness_weight": FAIRNESS_WEIGHT,
                "use_minmax_normalization": USE_MINMAX_NORMALIZATION,
                **test_result,
            })

    model_results_df = pd.DataFrame(model_results)
    model_results_df.to_csv(OUTPUT_MODEL_BREAKDOWN, index=False)

    print("\n=== Scalarized Claim 2 results by ML model ===")
    print(model_results_df)
    print(f"\n[SAVE] Model-level breakdown: {OUTPUT_MODEL_BREAKDOWN}")


    # ============================================================
    # Step 5: Paper-ready LaTeX table
    # ============================================================

    paper_df = global_results_df[
        [
            "comparison",
            "n_pairs",
            "mean_delta",
            "ci95_low",
            "ci95_high",
            "p_value",
            "win_rate",
            "tie_rate",
            "significant",
        ]
    ].copy()

    paper_df["95% CI"] = paper_df.apply(
        lambda r: f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]",
        axis=1,
    )

    paper_df["mean_delta"] = paper_df["mean_delta"].map(lambda x: f"{x:.4f}")
    paper_df["p_value"] = paper_df["p_value"].map(format_p_value)
    paper_df["win_rate"] = paper_df["win_rate"].map(lambda x: f"{100*x:.1f}\\%")
    paper_df["tie_rate"] = paper_df["tie_rate"].map(lambda x: f"{100*x:.1f}\\%")
    paper_df["significant"] = paper_df["significant"].map({True: "Yes", False: "No"})

    paper_df = paper_df[
        [
            "comparison",
            "n_pairs",
            "mean_delta",
            "95% CI",
            "p_value",
            "win_rate",
            "tie_rate",
            "significant",
        ]
    ]

    paper_df = paper_df.rename(columns={
        "comparison": "Comparison",
        "n_pairs": "$n$",
        "mean_delta": "$\\bar{d}$",
        "p_value": "$p$-value",
        "win_rate": "Win rate",
        "tie_rate": "Tie rate",
        "significant": "Significant",
    })

    latex_table = paper_df.to_latex(
        index=False,
        escape=False,
        column_format="lrrrrrrr",
        caption=(
            "Paired Wilcoxon signed-rank analysis for Claim~2 using a scalarized "
            "fairness--utility trade-off score under AIM across the four main datasets "
            "and three classifiers. The analysis is restricted to EOD/SPD-oriented "
            "trade-offs, and POST is defined as the best score among ROC and EqOdds. "
            "The scalarized score is "
            "$S=\\sqrt{w_U(1-U)^2+w_F|m|^2}$ with "
            f"$w_U={UTILITY_WEIGHT}$ and $w_F={FAIRNESS_WEIGHT}$. "
            "Negative $\\bar{d}$ indicates that POST has a lower, hence better, "
            "scalarized trade-off score than the compared stage."
        ),
        label="tab:claim2-scalarized-wilcoxon-all-models",
    )

    print("\n=== Paper-ready scalarized Claim 2 table ===")
    print(latex_table)

    with open(OUTPUT_LATEX_TABLE, "w") as f:
        f.write(latex_table)

    print(f"\n[SAVE] LaTeX table: {OUTPUT_LATEX_TABLE}")