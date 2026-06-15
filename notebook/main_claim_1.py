


import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, t


# ============================================================
# Claim 1: DP+Fair vs DP-only on fairness metrics
# Main datasets: Adult, ACSIncome/Folktables, COMPAS, BoD Config 5
# Setting: AIM synthesizer + all classifiers: XGB, RF, LR
#
# Expected folder structure:
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
# If automatic root detection fails, define manually before this cell:
#
# RESULTS_BASE_DIR = r"C:\Users\heber\OneDrive\Documents\5_ISFP_Inria\2_Master_Students\2025_Vinicius\DP-Benchmark"
#
# Hypothesis:
#   d_i = |m_DP+Fair_i| - |m_DP-only_i|
#
#   H0: median(d_i) = 0
#   H1: median(d_i) < 0
#
# Negative d_i means DP+Fair improves fairness over DP-only.
# ============================================================


# -------------------------
# Configuration
# -------------------------

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

FAIRNESS_METRICS = ["MAD", "EOD", "SPD"]

INTERVENTION_MAP = {
    "PRE": ["dir", "lfr", "reweigh"],
    "IN": ["egr", "gsr"],
    "POST": ["ceop", "eqodds", "roc"],
}

FAIR_METHODS = (
    INTERVENTION_MAP["PRE"]
    + INTERVENTION_MAP["IN"]
    + INTERVENTION_MAP["POST"]
)

METHOD_STAGE = {
    method: stage
    for stage, methods in INTERVENTION_MAP.items()
    for method in methods
}

ALPHA = 0.05

OUTPUT_DETAILED_DELTAS = (
    f"claim1_dp_fair_vs_dp_only_deltas_all_models_{DP_SYNTHESIZER}.csv"
)

OUTPUT_DATASET_METRIC_RESULTS = (
    f"claim1_dp_fair_vs_dp_only_by_dataset_metric_all_models_{DP_SYNTHESIZER}.csv"
)

OUTPUT_GLOBAL_METRIC_RESULTS = (
    f"claim1_dp_fair_vs_dp_only_global_by_metric_all_models_{DP_SYNTHESIZER}.csv"
)

OUTPUT_MODEL_METRIC_RESULTS = (
    f"claim1_dp_fair_vs_dp_only_by_model_metric_{DP_SYNTHESIZER}.csv"
)

OUTPUT_METHOD_RESULTS = (
    f"claim1_dp_fair_vs_dp_only_by_dataset_method_metric_all_models_{DP_SYNTHESIZER}.csv"
)

OUTPUT_SUMMARY_BY_METHOD = (
    f"claim1_summary_by_method_all_models_{DP_SYNTHESIZER}.csv"
)

OUTPUT_SUMMARY_BY_STAGE = (
    f"claim1_summary_by_stage_all_models_{DP_SYNTHESIZER}.csv"
)

OUTPUT_PAPER_TABLE_TEX = (
    f"claim1_wilcoxon_global_paper_table_all_models_{DP_SYNTHESIZER}.tex"
)

OUTPUT_PAPER_TABLE_CSV = (
    f"claim1_wilcoxon_global_paper_table_all_models_{DP_SYNTHESIZER}.csv"
)


# -------------------------
# Path helper functions
# -------------------------

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
            [
                f"BoD-{bod_combo}",
                f"BOD-{bod_combo}",
                f"bod-{bod_combo}",
                f"BoD_{bod_combo}",
                f"BOD_{bod_combo}",
                f"bod_{bod_combo}",
            ],
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
            [
                f"BoD-{bod_combo}",
                f"BOD-{bod_combo}",
                f"bod-{bod_combo}",
                f"BoD_{bod_combo}",
                f"BOD_{bod_combo}",
                f"bod_{bod_combo}",
            ],
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
        [
            path_dataset,
            path_dataset.lower(),
            path_dataset.upper(),
            "ACSIncome" if path_dataset == "ACSIncome" else path_dataset,
            "acs_income" if path_dataset == "ACSIncome" else path_dataset,
            "COMPAS" if path_dataset == "Compas" else path_dataset,
        ],
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
        [
            path_dataset,
            path_dataset.lower(),
            path_dataset.upper(),
            "ACSIncome" if path_dataset == "ACSIncome" else path_dataset,
            "acs_income" if path_dataset == "ACSIncome" else path_dataset,
            "COMPAS" if path_dataset == "Compas" else path_dataset,
        ],
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


# -------------------------
# Helper functions
# -------------------------

def detect_seed_column(df):
    possible_seed_columns = [
        "Seed", "seed", "RandomSeed", "random_seed", "run", "Run"
    ]

    for col in possible_seed_columns:
        if col in df.columns:
            return col

    raise ValueError(
        "No seed column found. The Wilcoxon signed-rank test requires "
        "per-seed values. Expected one of: "
        f"{possible_seed_columns}"
    )


def mean_ci95(values):
    """
    Computes the mean and 95% confidence interval using Student's t-distribution.

    mean +/- t_{0.975,n-1} * s / sqrt(n)
    """

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

    ci_low = float(mean - half_width)
    ci_high = float(mean + half_width)

    return mean, ci_low, ci_high, n


def run_one_sided_wilcoxon_less(deltas):
    """
    Runs a one-sided Wilcoxon signed-rank test.

    We test:
        H0: median(d_i) = 0
        H1: median(d_i) < 0

    where:
        d_i = |m_DP+Fair_i| - |m_DP-only_i|

    Negative d_i means DP+Fair improves fairness.
    """

    d = np.asarray(deltas, dtype=float)
    d = d[~np.isnan(d)]

    if len(d) < 2:
        return None

    mean_delta, ci_low, ci_high, n = mean_ci95(d)

    median_delta = float(np.median(d))
    win_rate = float(np.mean(d < 0))
    loss_rate = float(np.mean(d > 0))
    zero_rate = float(np.mean(d == 0))

    if np.allclose(d, 0):
        return {
            "n_pairs": n,
            "mean_delta": mean_delta,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "median_delta": median_delta,
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "zero_rate": zero_rate,
            "wilcoxon_stat": np.nan,
            "p_value": 1.0,
            "significant": False,
            "improves_fairness": False,
        }

    try:
        stat, p_value = wilcoxon(
            d,
            alternative="less",
            zero_method="wilcox",
            mode="auto",
        )
    except ValueError:
        return None

    return {
        "n_pairs": n,
        "mean_delta": mean_delta,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "median_delta": median_delta,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "zero_rate": zero_rate,
        "wilcoxon_stat": float(stat),
        "p_value": float(p_value),
        "significant": bool((p_value < ALPHA) and (mean_delta < 0)),
        "improves_fairness": bool(mean_delta < 0),
    }


def format_p_value(p):
    if pd.isna(p):
        return "NA"
    if p == 0 or p < 1e-300:
        return r"$<10^{-300}$"
    if p < 1e-3:
        return rf"${p:.2e}$"
    return f"{p:.3f}"


def prepare_metric_frame(df, seed_col, metric, dataset_key, dataset_display, ml_model):
    """
    Prepares paired DP-only and DP+Fair values for a given fairness metric.

    Pairing is performed by:
        seed, epsilon

    For each DP+Fair method, we compare against the DP-only result with
    the same seed and epsilon.
    """

    dp_only = df[
        df["Fair-Method"].isna()
        & df["Epsilon"].notna()
    ][[seed_col, "Epsilon", metric]].copy()

    dp_only = (
        dp_only
        .groupby([seed_col, "Epsilon"], dropna=False)[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: "dp_only"})
    )

    dp_fair = df[
        df["Fair-Method"].isin(FAIR_METHODS)
        & df["Epsilon"].notna()
    ][[seed_col, "Epsilon", "Fair-Method", metric]].copy()

    dp_fair = (
        dp_fair
        .groupby([seed_col, "Epsilon", "Fair-Method"], dropna=False)[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: "dp_fair"})
    )

    paired = pd.merge(
        dp_fair,
        dp_only,
        on=[seed_col, "Epsilon"],
        how="inner",
    )

    paired = paired.dropna(subset=["dp_fair", "dp_only"]).copy()

    paired = paired.rename(columns={seed_col: "seed"})

    paired["ml_model"] = ml_model
    paired["dataset"] = dataset_key
    paired["dataset_display"] = dataset_display
    paired["metric"] = metric
    paired["stage"] = paired["Fair-Method"].map(METHOD_STAGE)

    paired["abs_dp_fair"] = np.abs(paired["dp_fair"])
    paired["abs_dp_only"] = np.abs(paired["dp_only"])

    paired["delta"] = paired["abs_dp_fair"] - paired["abs_dp_only"]

    return paired


# -------------------------
# Load data and build paired differences
# -------------------------

RESULTS_ROOT = resolve_results_root()

print(f"[RESULTS_ROOT] {RESULTS_ROOT}")
print(f"[DP_SYNTHESIZER] {DP_SYNTHESIZER}")

all_delta_frames = []

for ml_model in ML_MODELS:
    print("\n" + "=" * 80)
    print(f"Reading Claim 1 data for ML model: {ml_model}")
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

        required_columns = ["Fair-Method", "Epsilon", seed_col] + FAIRNESS_METRICS
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Missing required columns for "
                f"ML={ml_model}, Dataset={dataset_display}: {missing_columns}"
            )

        unique_seeds = sorted(df[seed_col].dropna().unique())
        unique_eps = sorted(df["Epsilon"].dropna().unique())

        print(
            f"       ML={ml_model}, Dataset={dataset_display}, "
            f"n_seeds={len(unique_seeds)}, epsilons={unique_eps}"
        )

        for metric in FAIRNESS_METRICS:
            paired_metric_df = prepare_metric_frame(
                df=df,
                seed_col=seed_col,
                metric=metric,
                dataset_key=dataset_key,
                dataset_display=dataset_display,
                ml_model=ml_model,
            )

            if paired_metric_df.empty:
                print(
                    f"[SKIP] No paired data for "
                    f"ML={ml_model}, Dataset={dataset_display}, metric={metric}"
                )
                continue

            all_delta_frames.append(paired_metric_df)

if len(all_delta_frames) == 0:
    raise ValueError("No paired DP+Fair vs DP-only data found.")

deltas_df = pd.concat(all_delta_frames, ignore_index=True)

deltas_df.to_csv(OUTPUT_DETAILED_DELTAS, index=False)
print(f"\n[SAVE] Detailed paired deltas: {OUTPUT_DETAILED_DELTAS}")


# -------------------------
# Test A: dataset-level tests per fairness metric, pooled over models
# -------------------------

dataset_metric_results = []

for dataset_display in deltas_df["dataset_display"].unique():
    for metric in FAIRNESS_METRICS:
        sub = deltas_df[
            (deltas_df["dataset_display"] == dataset_display)
            & (deltas_df["metric"] == metric)
        ].copy()

        if sub.empty:
            continue

        test_result = run_one_sided_wilcoxon_less(sub["delta"].to_numpy())

        if test_result is None:
            print(f"[SKIP] Not enough paired data for {dataset_display}, {metric}")
            continue

        dataset_metric_results.append({
            "dataset": dataset_display,
            "dp_synthesizer": DP_SYNTHESIZER,
            "ml_models": ",".join(ML_MODELS),
            "metric": metric,
            "comparison": "DP+Fair_vs_DP-only",
            "n_unique_models": sub["ml_model"].nunique(),
            "n_unique_model_seed_dataset_triples": (
                sub[["ml_model", "dataset_display", "seed"]]
                .drop_duplicates()
                .shape[0]
            ),
            "n_unique_epsilons": sub["Epsilon"].nunique(),
            "n_methods": sub["Fair-Method"].nunique(),
            **test_result,
        })

dataset_metric_results_df = pd.DataFrame(dataset_metric_results)
dataset_metric_results_df.to_csv(OUTPUT_DATASET_METRIC_RESULTS, index=False)
print(f"[SAVE] Dataset-metric Wilcoxon results: {OUTPUT_DATASET_METRIC_RESULTS}")


# -------------------------
# Test B: global tests per fairness metric across datasets and models
# -------------------------

global_metric_results = []

for metric in FAIRNESS_METRICS:
    sub = deltas_df[deltas_df["metric"] == metric].copy()

    if sub.empty:
        continue

    test_result = run_one_sided_wilcoxon_less(sub["delta"].to_numpy())

    if test_result is None:
        print(f"[SKIP] Not enough paired data for global metric={metric}")
        continue

    global_metric_results.append({
        "dataset": "All main datasets",
        "dp_synthesizer": DP_SYNTHESIZER,
        "ml_models": ",".join(ML_MODELS),
        "metric": metric,
        "comparison": "DP+Fair_vs_DP-only",
        "n_unique_models": sub["ml_model"].nunique(),
        "n_unique_datasets": sub["dataset_display"].nunique(),
        "n_unique_model_dataset_seed_triples": (
            sub[["ml_model", "dataset_display", "seed"]]
            .drop_duplicates()
            .shape[0]
        ),
        "n_unique_epsilons": sub["Epsilon"].nunique(),
        "n_methods": sub["Fair-Method"].nunique(),
        **test_result,
    })

global_metric_results_df = pd.DataFrame(global_metric_results)
global_metric_results_df.to_csv(OUTPUT_GLOBAL_METRIC_RESULTS, index=False)
print(f"[SAVE] Global metric Wilcoxon results: {OUTPUT_GLOBAL_METRIC_RESULTS}")


# -------------------------
# Test C: model-level tests per fairness metric
# -------------------------

model_metric_results = []

for ml_model in sorted(deltas_df["ml_model"].unique()):
    for metric in FAIRNESS_METRICS:
        sub = deltas_df[
            (deltas_df["ml_model"] == ml_model)
            & (deltas_df["metric"] == metric)
        ].copy()

        if sub.empty:
            continue

        test_result = run_one_sided_wilcoxon_less(sub["delta"].to_numpy())

        if test_result is None:
            print(f"[SKIP] Not enough paired data for {ml_model}, {metric}")
            continue

        model_metric_results.append({
            "ml_model": ml_model,
            "dp_synthesizer": DP_SYNTHESIZER,
            "metric": metric,
            "comparison": "DP+Fair_vs_DP-only",
            "n_unique_datasets": sub["dataset_display"].nunique(),
            "n_unique_seed_dataset_pairs": (
                sub[["dataset_display", "seed"]]
                .drop_duplicates()
                .shape[0]
            ),
            "n_unique_epsilons": sub["Epsilon"].nunique(),
            "n_methods": sub["Fair-Method"].nunique(),
            **test_result,
        })

model_metric_results_df = pd.DataFrame(model_metric_results)
model_metric_results_df.to_csv(OUTPUT_MODEL_METRIC_RESULTS, index=False)
print(f"[SAVE] Model-metric Wilcoxon results: {OUTPUT_MODEL_METRIC_RESULTS}")


# -------------------------
# Test D: method-level tests per dataset, model, and metric
# -------------------------

method_results = []

for ml_model in sorted(deltas_df["ml_model"].unique()):
    for dataset_display in deltas_df["dataset_display"].unique():
        for metric in FAIRNESS_METRICS:
            for method in FAIR_METHODS:
                sub = deltas_df[
                    (deltas_df["ml_model"] == ml_model)
                    & (deltas_df["dataset_display"] == dataset_display)
                    & (deltas_df["metric"] == metric)
                    & (deltas_df["Fair-Method"] == method)
                ].copy()

                if sub.empty:
                    continue

                test_result = run_one_sided_wilcoxon_less(sub["delta"].to_numpy())

                if test_result is None:
                    print(
                        f"[SKIP] Not enough paired data for "
                        f"{ml_model}, {dataset_display}, {metric}, {method}"
                    )
                    continue

                method_results.append({
                    "ml_model": ml_model,
                    "dataset": dataset_display,
                    "dp_synthesizer": DP_SYNTHESIZER,
                    "metric": metric,
                    "method": method,
                    "stage": METHOD_STAGE[method],
                    "comparison": "DP+Fair_vs_DP-only",
                    "n_unique_seeds": sub["seed"].nunique(),
                    "n_unique_epsilons": sub["Epsilon"].nunique(),
                    **test_result,
                })

method_results_df = pd.DataFrame(method_results)
method_results_df.to_csv(OUTPUT_METHOD_RESULTS, index=False)
print(f"[SAVE] Method-level Wilcoxon results: {OUTPUT_METHOD_RESULTS}")


# -------------------------
# Summary tables
# -------------------------

print("\n==============================")
print("CLAIM 1: DP+Fair vs DP-only")
print("==============================")

print("\nHypothesis:")
print("d_i = |m_DP+Fair_i| - |m_DP-only_i|")
print("H0: median(d_i) = 0")
print("H1: median(d_i) < 0")
print("Negative d_i means DP+Fair improves fairness.\n")

print("=== Dataset-level results per fairness metric, pooled over models ===")
print(dataset_metric_results_df)

print("\n=== Global results per fairness metric across datasets and models ===")
print(global_metric_results_df)

print("\n=== Model-level results per fairness metric ===")
print(model_metric_results_df)


if not method_results_df.empty:
    print("\n=== Summary by method across all datasets and models ===")

    summary_by_method = (
        method_results_df
        .groupby(["stage", "method"])
        .agg(
            total_tests=("significant", "count"),
            significant_improvements=("significant", "sum"),
            median_mean_delta=("mean_delta", "median"),
            median_p_value=("p_value", "median"),
            mean_win_rate=("win_rate", "mean"),
        )
        .reset_index()
    )

    summary_by_method["percentage_significant"] = (
        100
        * summary_by_method["significant_improvements"]
        / summary_by_method["total_tests"]
    )

    print(summary_by_method)
    summary_by_method.to_csv(OUTPUT_SUMMARY_BY_METHOD, index=False)
    print(f"[SAVE] Summary by method: {OUTPUT_SUMMARY_BY_METHOD}")

    print("\n=== Summary by stage across all datasets and models ===")

    summary_by_stage = (
        method_results_df
        .groupby("stage")
        .agg(
            total_tests=("significant", "count"),
            significant_improvements=("significant", "sum"),
            median_mean_delta=("mean_delta", "median"),
            median_p_value=("p_value", "median"),
            mean_win_rate=("win_rate", "mean"),
        )
        .reset_index()
    )

    summary_by_stage["percentage_significant"] = (
        100
        * summary_by_stage["significant_improvements"]
        / summary_by_stage["total_tests"]
    )

    print(summary_by_stage)
    summary_by_stage.to_csv(OUTPUT_SUMMARY_BY_STAGE, index=False)
    print(f"[SAVE] Summary by stage: {OUTPUT_SUMMARY_BY_STAGE}")


# -------------------------
# Paper-ready global table
# -------------------------

paper_df = global_metric_results_df[
    [
        "metric",
        "n_pairs",
        "mean_delta",
        "ci95_low",
        "ci95_high",
        "p_value",
        "win_rate",
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
paper_df["significant"] = paper_df["significant"].map({True: "Yes", False: "No"})

paper_df = paper_df[
    [
        "metric",
        "n_pairs",
        "mean_delta",
        "95% CI",
        "p_value",
        "win_rate",
        "significant",
    ]
]

paper_df = paper_df.rename(columns={
    "metric": "Metric",
    "n_pairs": "$n$",
    "mean_delta": "$\\bar{d}$",
    "p_value": "$p$-value",
    "win_rate": "Win rate",
    "significant": "Significant",
})

paper_df.to_csv(OUTPUT_PAPER_TABLE_CSV, index=False)

latex_table = paper_df.to_latex(
    index=False,
    escape=False,
    column_format="lrrrrrr",
    caption=(
        "Global paired Wilcoxon signed-rank analysis for Claim~1 under AIM "
        "across the four main datasets and three classifiers. "
        "For each fairness metric, we compare \\dpfair{} against \\dponly{} using "
        "$d_i=|m^{(i)}_{\\mathrm{DP+Fair}}|-|m^{(i)}_{\\mathrm{DP-only}}|$. "
        "Negative $\\bar{d}$ indicates that \\dpfair{} reduces the absolute fairness gap "
        "relative to \\dponly{}."
    ),
    label="tab:claim1-wilcoxon-global-all-models",
)

with open(OUTPUT_PAPER_TABLE_TEX, "w", encoding="utf-8") as f:
    f.write(latex_table)

print(f"[SAVE] Paper-ready CSV table: {OUTPUT_PAPER_TABLE_CSV}")
print(f"[SAVE] Paper-ready LaTeX table: {OUTPUT_PAPER_TABLE_TEX}")

print("\n=== Paper-ready Claim 1 table ===")
print(latex_table)








