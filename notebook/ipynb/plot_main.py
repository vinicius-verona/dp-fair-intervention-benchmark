




import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from scipy.stats import t

params = {'axes.titlesize':'18',
          'xtick.labelsize':'17',
          'ytick.labelsize':'17',
          'font.size':'17',
          'legend.fontsize':'medium',
          'lines.linewidth':'2.0',
          'font.weight':'normal',
          'lines.markersize':'8',
          'text.latex.preamble': r'\usepackage{amsfonts}',
          }
matplotlib.rcParams.update(params)
plt.rcParams["mathtext.fontset"] = "cm"
plt.rc('text', usetex=True)
plt.rc('font', family='serif')



# =========================
# Configuration
# =========================
DP_SYNTHETIZER = 'aim'   # choose among: 'mst', 'aim'
INTERVENTION = 'all'     # choose among: 'all', 'PRE', 'IN', 'POST'

ML_MODELS = ['LR', 'RF', 'XGB'] # 
UTILITY_METRICS = ['ACC', 'F1']
FAIRNESS_METRICS = ['MAD', 'EOD', 'SPD']

METRIC_LABELS = {
    'ACC': 'Accuracy (ACC)',
    'PREC': 'Precision (PREC)',
    'REC': 'Recall (REC)',
    'F1': 'F1-score (F1)',
    'MAD': 'Model Accuracy Difference (MAD)',
    'EOD': 'Equalized Opportunity Difference (EOD)',
    'SPD': 'Statistical Parity Difference (SPD)'
}

METHOD_ACRONYMS = {
    'reweigh': 'RW',
    'dir': 'DIR',
    'lfr': 'LFR',
    'egr': 'EGR',
    'gsr': 'GSR',
    'roc': 'ROC',
    'eqodds': 'EQODDS',
    'ceop': 'CEOP',
    'DP-Only': 'DP-ONLY'
}

DATASETS = ['adult', 'folktables', 'compas', 'bod']
AVERAGE_BY_SEED = True
DP_ONLY_LABEL = 'DP-ONLY'

# =========================
# Categorization
# =========================
intervention_map = {
    'PRE': ['dir', 'lfr', 'reweigh'],
    'IN': ['egr', 'gsr'],
    'POST': ['ceop', 'eqodds', 'roc']
}

method_to_cat = {m: cat for cat, ms in intervention_map.items() for m in ms}
method_to_cat[DP_ONLY_LABEL] = 'BASE'

if INTERVENTION.lower() == 'all':
    fair_methods = intervention_map['PRE'] + intervention_map['IN'] + intervention_map['POST']
elif INTERVENTION.upper() in intervention_map:
    fair_methods = intervention_map[INTERVENTION.upper()]
else:
    raise ValueError("INTERVENTION must be one of: 'all', 'PRE', 'IN', 'POST'.")

all_methods = fair_methods + [DP_ONLY_LABEL]

# =========================
# Visual mapping
# =========================
universe_methods = intervention_map['PRE'] + intervention_map['IN'] + intervention_map['POST'] + [DP_ONLY_LABEL]
colors_map = {m: plt.cm.tab10(i % 10) for i, m in enumerate(universe_methods)}
colors_map[DP_ONLY_LABEL] = 'magenta'

markers_map = {
    'dir': 'o', 'lfr': 'o', 'reweigh': 'o',
    'egr': 's', 'gsr': 's',
    'ceop': '^', 'eqodds': '^', 'roc': '^',
    DP_ONLY_LABEL: 'X'
}

# =========================
# Intervention stage mapping
# =========================
method_to_stage = {
    m: stage for stage, methods in intervention_map.items() for m in methods
}

STAGE_ORDER = ['DP-ONLY', 'PRE', 'IN', 'POST']

STAGE_COLORS = {
    'DP-ONLY': 'magenta',
    'PRE': '#1f77b4',
    'IN': '#ff7f0e',
    'POST': '#2ca02c'
}

STAGE_MARKERS = {
    'DP-ONLY': 'X',
    'PRE': 'o',
    'IN': 's',
    'POST': '^'
}

BASELINE_LABEL = 'BASELINE'
DP_ONLY_LABEL  = 'DP-ONLY'
FAIR_ONLY_LABEL = r'FAIR-ONLY ($\varepsilon=\infty$)'



# ============================================================
# Compas + ACSIncome + Adult + BoD result plotting code
#
# Expected folder structure:
#
# root/
#   Compas/
#     output/<ML_MODEL>/Compas/<DP_SYNTHETIZER>/results/*.csv
#
#   ACSIncome/
#     output/<ML_MODEL>/ACSIncome/<DP_SYNTHETIZER>/results/*.csv
#
#   Adult/
#     output/<ML_MODEL>/Adult/<DP_SYNTHETIZER>/results/*.csv
#
#   BoD/
#     BoD-1/
#       output/<ML_MODEL>/BoD-1/<DP_SYNTHETIZER>/results/*.csv
#     BoD-2/
#       output/<ML_MODEL>/BoD-2/<DP_SYNTHETIZER>/results/*.csv
#     ...
#     BoD-6/
#       output/<ML_MODEL>/BoD-6/<DP_SYNTHETIZER>/results/*.csv
# ============================================================


# -----------------------------
# Main settings
# -----------------------------

DATASETS = [
    "Compas",
    "ACSIncome",
    "Adult",
    "BoD-1",
    "BoD-2",
    "BoD-3",
    "BoD-4",
    "BoD-5",
    "BoD-6",
]


SHOW_CI = True
CI_METHODS = "all"

MAX_X_CI = 0.20
MAX_Y_CI = 0.05

omitted_ci_rows = []


# -----------------------------
# Path helpers
# -----------------------------

def _normalize_dataset_name(dataset):
    dataset_str = str(dataset)

    aliases = {
        "compas": "Compas",
        "adult": "Adult",
        "acsincome": "ACSIncome",
        "acs_income": "ACSIncome",
        "acs-income": "ACSIncome",
        "folktables": "ACSIncome",
        "bod": "BoD",
    }

    lower = dataset_str.lower()

    if lower.startswith("bod-"):
        try:
            combo = int(lower.split("-", 1)[1])
            return f"BoD-{combo}"
        except Exception:
            return dataset_str

    return aliases.get(lower, dataset_str)


def _is_bod_dataset(dataset):
    normalized = _normalize_dataset_name(dataset)
    return str(normalized).lower().startswith("bod-")


def _bod_combo_from_dataset(dataset):
    normalized = _normalize_dataset_name(dataset)

    if not _is_bod_dataset(normalized):
        return None

    try:
        return int(str(normalized).split("-", 1)[1])
    except Exception:
        raise ValueError(
            f"Invalid BoD dataset name: {dataset}. "
            "Expected format: BoD-1, BoD-2, ..., BoD-6."
        )


def _dataset_folder_candidates(dataset):
    normalized = _normalize_dataset_name(dataset)

    if normalized == "Compas":
        return ["Compas", "COMPAS", "compas"]

    if normalized == "Adult":
        return ["Adult", "ADULT", "adult"]

    if normalized == "ACSIncome":
        return [
            "ACSIncome",
            "ACSincome",
            "acsincome",
            "ACS_Income",
            "acs_income",
        ]

    if _is_bod_dataset(normalized):
        combo = _bod_combo_from_dataset(normalized)
        return [
            f"BoD-{combo}",
            f"BOD-{combo}",
            f"bod-{combo}",
            f"BoD_{combo}",
            f"BOD_{combo}",
            f"bod_{combo}",
        ]

    return [normalized, str(dataset)]


def _find_case_insensitive_dir(parent_dir, candidates):
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


def _resolve_results_root():
    """
    Finds the root directory that contains at least one dataset folder.

    Expected structure:

        root/
            Compas/
            ACSIncome/
            Adult/
            BoD/

    Optional manual override:

        RESULTS_BASE_DIR = r"/path/to/root"

    or any dataset-specific folder:

        RESULTS_BASE_DIR = r"/path/to/root/Compas"
        RESULTS_BASE_DIR = r"/path/to/root/ACSIncome"
        RESULTS_BASE_DIR = r"/path/to/root/Adult"
        RESULTS_BASE_DIR = r"/path/to/root/BoD"
        RESULTS_BASE_DIR = r"/path/to/root/BoD/BoD-5"
    """

    top_level_dataset_names = [
        "Compas",
        "ACSIncome",
        "Adult",
        "BoD",
    ]

    if "RESULTS_BASE_DIR" in globals():
        candidate = os.path.abspath(os.path.expanduser(str(RESULTS_BASE_DIR)))

        # Case 1: RESULTS_BASE_DIR points to the root folder.
        for dataset_name in top_level_dataset_names:
            if os.path.isdir(os.path.join(candidate, dataset_name)):
                return candidate

        # Case 2: RESULTS_BASE_DIR points directly to root/<DATASET>.
        base = os.path.basename(candidate).lower()

        if base in [name.lower() for name in top_level_dataset_names]:
            return os.path.dirname(candidate)

        # Case 3: RESULTS_BASE_DIR points to root/BoD/BoD-i.
        if base.startswith("bod-"):
            parent = os.path.dirname(candidate)
            grandparent = os.path.dirname(parent)

            if os.path.basename(parent).lower() == "bod":
                return grandparent

    cwd = os.path.abspath(os.getcwd())

    candidates = []

    parent = cwd
    while True:
        candidates.append(parent)

        new_parent = os.path.dirname(parent)

        if new_parent == parent:
            break

        parent = new_parent

    for candidate in candidates:
        for dataset_name in top_level_dataset_names:
            if os.path.isdir(os.path.join(candidate, dataset_name)):
                return candidate

    raise FileNotFoundError(
        "Could not find Compas, ACSIncome, Adult, or BoD folder.\n\n"
        f"Current working directory:\n{cwd}\n\n"
        "Expected structure:\n"
        "root/Compas/output/<ML_MODEL>/Compas/<DP_SYNTHETIZER>/results/\n"
        "root/ACSIncome/output/<ML_MODEL>/ACSIncome/<DP_SYNTHETIZER>/results/\n"
        "root/Adult/output/<ML_MODEL>/Adult/<DP_SYNTHETIZER>/results/\n"
        "root/BoD/BoD-5/output/<ML_MODEL>/BoD-5/<DP_SYNTHETIZER>/results/\n\n"
        "You can define RESULTS_BASE_DIR manually, for example:\n"
        "RESULTS_BASE_DIR = r'C:\\path\\to\\root'\n"
        "or:\n"
        "RESULTS_BASE_DIR = r'C:\\path\\to\\root\\BoD'\n"
        "or:\n"
        "RESULTS_BASE_DIR = r'C:\\path\\to\\root\\BoD\\BoD-5'"
    )


def _get_results_dir(results_root, ML_MODEL, dataset, DP_SYNTHETIZER):
    normalized_dataset = _normalize_dataset_name(dataset)

    model_candidates = [
        str(ML_MODEL),
        str(ML_MODEL).lower(),
        str(ML_MODEL).upper(),
    ]

    synth_candidates = [
        str(DP_SYNTHETIZER),
        str(DP_SYNTHETIZER).lower(),
        str(DP_SYNTHETIZER).upper(),
    ]

    # --------------------------------------------------------
    # Special case: BoD
    #
    # root/BoD/BoD-5/output/XGB/BoD-5/aim/results/
    # --------------------------------------------------------
    if _is_bod_dataset(normalized_dataset):
        combo = _bod_combo_from_dataset(normalized_dataset)

        # root/BoD
        bod_root_dir = _find_case_insensitive_dir(
            results_root,
            ["BoD", "BOD", "bod"],
        )
        if bod_root_dir is None:
            return None

        # root/BoD/BoD-5
        bod_combo_dir = _find_case_insensitive_dir(
            bod_root_dir,
            [
                f"BoD-{combo}",
                f"BOD-{combo}",
                f"bod-{combo}",
                f"BoD_{combo}",
                f"BOD_{combo}",
                f"bod_{combo}",
            ],
        )
        if bod_combo_dir is None:
            return None

        # root/BoD/BoD-5/output
        output_dir = _find_case_insensitive_dir(
            bod_combo_dir,
            ["output", "Output"],
        )
        if output_dir is None:
            return None

        # root/BoD/BoD-5/output/XGB
        model_dir = _find_case_insensitive_dir(
            output_dir,
            model_candidates,
        )
        if model_dir is None:
            return None

        # root/BoD/BoD-5/output/XGB/BoD-5
        dataset_dir_2 = _find_case_insensitive_dir(
            model_dir,
            [
                f"BoD-{combo}",
                f"BOD-{combo}",
                f"bod-{combo}",
                f"BoD_{combo}",
                f"BOD_{combo}",
                f"bod_{combo}",
            ],
        )
        if dataset_dir_2 is None:
            return None

        # root/BoD/BoD-5/output/XGB/BoD-5/aim
        synth_dir = _find_case_insensitive_dir(
            dataset_dir_2,
            synth_candidates,
        )
        if synth_dir is None:
            return None

        # root/BoD/BoD-5/output/XGB/BoD-5/aim/results
        results_dir = _find_case_insensitive_dir(
            synth_dir,
            ["results", "Results"],
        )

        return results_dir

    # --------------------------------------------------------
    # Regular datasets:
    #
    # root/<DATASET>/output/<ML_MODEL>/<DATASET>/<SYNTH>/results/
    # --------------------------------------------------------

    dataset_candidates = _dataset_folder_candidates(normalized_dataset)

    # root/<DATASET>
    dataset_dir_1 = _find_case_insensitive_dir(
        results_root,
        dataset_candidates,
    )
    if dataset_dir_1 is None:
        return None

    # root/<DATASET>/output
    output_dir = _find_case_insensitive_dir(
        dataset_dir_1,
        ["output", "Output"],
    )
    if output_dir is None:
        return None

    # root/<DATASET>/output/XGB, RF, LR, etc.
    model_dir = _find_case_insensitive_dir(
        output_dir,
        model_candidates,
    )
    if model_dir is None:
        return None

    # root/<DATASET>/output/XGB/<DATASET>
    dataset_dir_2 = _find_case_insensitive_dir(
        model_dir,
        dataset_candidates,
    )
    if dataset_dir_2 is None:
        return None

    # root/<DATASET>/output/XGB/<DATASET>/aim or mst
    synth_dir = _find_case_insensitive_dir(
        dataset_dir_2,
        synth_candidates,
    )
    if synth_dir is None:
        return None

    # root/<DATASET>/output/XGB/<DATASET>/aim/results
    results_dir = _find_case_insensitive_dir(
        synth_dir,
        ["results", "Results"],
    )

    return results_dir


# -----------------------------
# CSV loading helpers
# -----------------------------

def _extract_seed_from_filename(file_path):
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


def _find_seed_csvs(results_root, ML_MODEL, dataset, DP_SYNTHETIZER):
    synth_lower = str(DP_SYNTHETIZER).lower()

    results_dir = _get_results_dir(
        results_root=results_root,
        ML_MODEL=ML_MODEL,
        dataset=dataset,
        DP_SYNTHETIZER=DP_SYNTHETIZER,
    )

    if results_dir is None:
        return [], None

    csv_files = []

    for file_name in os.listdir(results_dir):
        file_path = os.path.join(results_dir, file_name)

        if not os.path.isfile(file_path):
            continue

        f_lower = file_name.lower()

        if (
            f_lower.startswith("benchmark_results_seeds_")
            and f_lower.endswith(".csv")
            and f"_synth_{synth_lower}" in f_lower
        ):
            csv_files.append(file_path)

    csv_files = sorted(csv_files, key=lambda p: os.path.basename(p))

    return csv_files, results_dir


def _read_seed_csvs(results_root, ML_MODEL, dataset, DP_SYNTHETIZER):
    csv_files, results_dir = _find_seed_csvs(
        results_root=results_root,
        ML_MODEL=ML_MODEL,
        dataset=dataset,
        DP_SYNTHETIZER=DP_SYNTHETIZER,
    )

    normalized_dataset = _normalize_dataset_name(dataset)

    if len(csv_files) == 0:
        print(
            f"[SKIP] No CSV files found for "
            f"ML_MODEL={ML_MODEL}, "
            f"dataset={dataset}, "
            f"normalized_dataset={normalized_dataset}, "
            f"synth={DP_SYNTHETIZER}, "
            f"RESULTS_ROOT={results_root}, "
            f"searched_dir={results_dir}"
        )
        return None

    dfs = []

    for csv_file in csv_files:
        tmp = pd.read_csv(csv_file)
        tmp = tmp.replace([np.inf, -np.inf], np.nan)

        if "Seed" not in tmp.columns:
            tmp["Seed"] = _extract_seed_from_filename(csv_file)

        tmp["Source-File"] = os.path.basename(csv_file)
        tmp["Source-Path"] = csv_file
        tmp["Dataset"] = normalized_dataset

        dfs.append(tmp)

    df = pd.concat(dfs, ignore_index=True)

    print(
        f"[READ] "
        f"{ML_MODEL} | "
        f"{dataset} -> {normalized_dataset} | "
        f"{DP_SYNTHETIZER}: "
        f"{len(csv_files)} CSV files, {len(df)} rows"
    )
    print(f"[DIR] {results_dir}")

    return df


# -----------------------------
# Statistical helpers
# -----------------------------

def _detect_seed_column(df):
    for col in ["Seed", "seed", "RandomSeed", "random_seed", "run", "Run"]:
        if col in df.columns:
            return col

    raise ValueError("No seed column found. CI requires per-seed results.")


def _mean_ci95(series):
    values = series.dropna().to_numpy(dtype=float)
    n = len(values)

    if n == 0:
        return pd.Series({
            "mean": np.nan,
            "ci95": np.nan,
            "n": 0,
        })

    if n == 1:
        return pd.Series({
            "mean": values[0],
            "ci95": 0.0,
            "n": 1,
        })

    mean = np.mean(values)
    sem = np.std(values, ddof=1) / np.sqrt(n)
    ci95 = t.ppf(0.975, df=n - 1) * sem

    return pd.Series({
        "mean": mean,
        "ci95": ci95,
        "n": n,
    })


def _should_show_ci(method, xerr, yerr):
    if not SHOW_CI:
        return False

    if not (CI_METHODS == "all" or method in CI_METHODS):
        return False

    if not np.isfinite(xerr) or not np.isfinite(yerr):
        return False

    if xerr > MAX_X_CI or yerr > MAX_Y_CI:
        return False

    return True


if __name__ == "__main__":

    # -----------------------------
    # Resolve root
    # -----------------------------

    RESULTS_ROOT = _resolve_results_root()

    print(f"[RESULTS_ROOT] {RESULTS_ROOT}")
    print(f"[DP_SYNTHETIZER] {DP_SYNTHETIZER}")
    print(f"[DATASETS] {DATASETS}")
    print(f"[CWD] {os.getcwd()}")


    # -----------------------------
    # Main plotting loop
    # -----------------------------

    for ML_MODEL in ML_MODELS:
        for UTILITY_METRIC in UTILITY_METRICS:
            for dataset in DATASETS:

                normalized_dataset = _normalize_dataset_name(dataset)

                df = _read_seed_csvs(
                    results_root=RESULTS_ROOT,
                    ML_MODEL=ML_MODEL,
                    dataset=normalized_dataset,
                    DP_SYNTHETIZER=DP_SYNTHETIZER,
                )

                if df is None:
                    continue

                df = df.replace([np.inf, -np.inf], np.nan)

                seed_col = _detect_seed_column(df)

                if UTILITY_METRIC == "F1":
                    if not {"PREC", "REC"}.issubset(df.columns):
                        raise ValueError("F1 requires PREC and REC columns in the CSV.")

                    df["F1"] = np.where(
                        (df["PREC"] + df["REC"]) > 0,
                        2 * df["PREC"] * df["REC"] / (df["PREC"] + df["REC"]),
                        0.0,
                    )

                baseline_df = df[
                    df["Fair-Method"].isna()
                    & df["DP-Method"].isna()
                    & df["Epsilon"].isna()
                ].copy()

                dp_only_df = df[
                    df["Fair-Method"].isna()
                    & df["Epsilon"].notna()
                ].copy()

                dp_only_df["Fair-Method"] = DP_ONLY_LABEL

                fair_df = df[
                    df["Fair-Method"].isin(
                        intervention_map["PRE"]
                        + intervention_map["IN"]
                        + intervention_map["POST"]
                    )
                ].copy()

                combined_raw_df = pd.concat(
                    [fair_df, dp_only_df],
                    ignore_index=True,
                )

                grouped_rows = []

                for (method, eps), g in combined_raw_df.groupby(
                    ["Fair-Method", "Epsilon"],
                    dropna=False,
                ):
                    row = {
                        "Fair-Method": method,
                        "Epsilon": eps,
                    }

                    for metric in FAIRNESS_METRICS + [UTILITY_METRIC]:
                        stats = _mean_ci95(g[metric])
                        row[f"{metric}_mean"] = stats["mean"]
                        row[f"{metric}_ci95"] = stats["ci95"]
                        row[f"{metric}_n"] = stats["n"]

                    grouped_rows.append(row)

                combined_df = pd.DataFrame(grouped_rows)

                ds_baseline = None

                if not baseline_df.empty:
                    ds_baseline = {}

                    for metric in FAIRNESS_METRICS + [UTILITY_METRIC]:
                        stats = _mean_ci95(baseline_df[metric])
                        ds_baseline[f"{metric}_mean"] = stats["mean"]
                        ds_baseline[f"{metric}_ci95"] = stats["ci95"]
                        ds_baseline[f"{metric}_n"] = stats["n"]

                fig, axes = plt.subplots(1, 3, figsize=(22, 5.1))

                for c, FAIRNESS_METRIC in enumerate(FAIRNESS_METRICS):
                    ax = axes[c]

                    if ds_baseline is not None:
                        x = ds_baseline[f"{FAIRNESS_METRIC}_mean"]
                        y = ds_baseline[f"{UTILITY_METRIC}_mean"]
                        xerr = ds_baseline[f"{FAIRNESS_METRIC}_ci95"]
                        yerr = ds_baseline[f"{UTILITY_METRIC}_ci95"]

                        if _should_show_ci(BASELINE_LABEL, xerr, yerr):
                            ax.errorbar(
                                x,
                                y,
                                xerr=xerr,
                                yerr=yerr,
                                fmt="none",
                                ecolor="black",
                                elinewidth=1.0,
                                capsize=2,
                                alpha=0.45,
                                zorder=8,
                            )

                        elif SHOW_CI:
                            omitted_ci_rows.append({
                                "ml_model": ML_MODEL,
                                "utility_metric": UTILITY_METRIC,
                                "dataset": normalized_dataset,
                                "fairness_metric": FAIRNESS_METRIC,
                                "method": BASELINE_LABEL,
                                "epsilon": np.nan,
                                "x_ci95": xerr,
                                "y_ci95": yerr,
                                "reason": "CI exceeds threshold or invalid",
                            })

                        ax.scatter(
                            x,
                            y,
                            marker="*",
                            s=650,
                            color="black",
                            edgecolors="white",
                            linewidth=1.2,
                            zorder=12,
                        )

                    for method in all_methods:
                        data = combined_df[
                            combined_df["Fair-Method"] == method
                        ].copy()

                        if data.empty:
                            continue

                        data = data.sort_values(
                            "Epsilon",
                            key=lambda x: x.fillna(999),
                        )

                        x_col = f"{FAIRNESS_METRIC}_mean"
                        y_col = f"{UTILITY_METRIC}_mean"
                        xerr_col = f"{FAIRNESS_METRIC}_ci95"
                        yerr_col = f"{UTILITY_METRIC}_ci95"

                        if len(data) > 1:
                            ax.plot(
                                data[x_col],
                                data[y_col],
                                linestyle="--",
                                linewidth=1.2,
                                color=colors_map[method],
                                alpha=0.8,
                            )

                        for _, row in data.iterrows():
                            eps = row["Epsilon"]

                            size = (
                                320
                                if pd.isna(eps)
                                else 100 + (min(eps, 20) / 20) * 200
                            )

                            x = row[x_col]
                            y = row[y_col]
                            xerr = row[xerr_col]
                            yerr = row[yerr_col]

                            if _should_show_ci(method, xerr, yerr):
                                ax.errorbar(
                                    x,
                                    y,
                                    xerr=xerr,
                                    yerr=yerr,
                                    fmt="none",
                                    ecolor=colors_map[method],
                                    elinewidth=1.0,
                                    capsize=2,
                                    alpha=0.15,
                                    zorder=4,
                                )

                            elif SHOW_CI and (
                                CI_METHODS == "all"
                                or method in CI_METHODS
                            ):
                                omitted_ci_rows.append({
                                    "ml_model": ML_MODEL,
                                    "utility_metric": UTILITY_METRIC,
                                    "dataset": normalized_dataset,
                                    "fairness_metric": FAIRNESS_METRIC,
                                    "method": method,
                                    "epsilon": eps,
                                    "x_ci95": xerr,
                                    "y_ci95": yerr,
                                    "reason": "CI exceeds threshold or invalid",
                                })

                            if pd.isna(eps):
                                ax.scatter(
                                    x,
                                    y,
                                    marker=markers_map[method],
                                    s=size,
                                    facecolors="white",
                                    edgecolors=colors_map[method],
                                    linewidths=2.5,
                                    zorder=10,
                                )

                            else:
                                ax.scatter(
                                    x,
                                    y,
                                    marker=markers_map[method],
                                    s=size,
                                    color=colors_map[method],
                                    edgecolors="k",
                                    alpha=1,
                                    zorder=9,
                                )

                    ax.set_xlabel(METRIC_LABELS[FAIRNESS_METRIC])

                    if c == 0:
                        ax.set_ylabel(METRIC_LABELS[UTILITY_METRIC])

                    ax.grid(True, linestyle=":", alpha=0.6)

                legend_items = [
                    plt.Line2D(
                        [0], [0],
                        marker="*",
                        linestyle="none",
                        markerfacecolor="black",
                        markeredgecolor="white",
                        markersize=22,
                        label=BASELINE_LABEL,
                    ),
                    plt.Line2D(
                        [0], [0],
                        marker="X",
                        linestyle="none",
                        markerfacecolor="magenta",
                        markeredgecolor="k",
                        markersize=14,
                        label=DP_ONLY_LABEL,
                    ),
                    plt.Line2D(
                        [0], [0],
                        marker="o",
                        linestyle="none",
                        markerfacecolor="white",
                        markeredgecolor="black",
                        markeredgewidth=2.5,
                        markersize=14,
                        label=FAIR_ONLY_LABEL,
                    ),
                ]

                if INTERVENTION.lower() in ["all", "pre"]:
                    for m in intervention_map["PRE"]:
                        legend_items.append(
                            plt.Line2D(
                                [0], [0],
                                marker="o",
                                linestyle="--",
                                color=colors_map[m],
                                markeredgecolor="k",
                                markersize=12,
                                label=f"PRE: {METHOD_ACRONYMS[m]}",
                            )
                        )

                if INTERVENTION.lower() in ["all", "in"]:
                    for m in intervention_map["IN"]:
                        legend_items.append(
                            plt.Line2D(
                                [0], [0],
                                marker="s",
                                linestyle="--",
                                color=colors_map[m],
                                markeredgecolor="k",
                                markersize=12,
                                label=f"IN: {METHOD_ACRONYMS[m]}",
                            )
                        )

                    legend_items.append(
                        plt.Line2D(
                            [0], [0],
                            linestyle="none",
                            alpha=0,
                            label="",
                        )
                    )

                if INTERVENTION.lower() in ["all", "post"]:
                    for m in intervention_map["POST"]:
                        legend_items.append(
                            plt.Line2D(
                                [0], [0],
                                marker="^",
                                linestyle="--",
                                color=colors_map[m],
                                markeredgecolor="k",
                                markersize=12,
                                label=f"POST: {METHOD_ACRONYMS[m]}",
                            )
                        )

                fig.legend(
                    handles=legend_items,
                    loc="upper center",
                    ncol=4 if INTERVENTION.lower() == "all" else 2,
                    bbox_to_anchor=(0.5, 1.13),
                )

                plt.tight_layout(rect=[0, 0, 1, 0.92])

                output_file = (
                    f"fig_results_"
                    f"{ML_MODEL}_{DP_SYNTHETIZER}_{UTILITY_METRIC}_"
                    f"{INTERVENTION}_{normalized_dataset}.pdf"
                )

                plt.savefig(
                    output_file,
                    dpi=500,
                    bbox_inches="tight",
                    pad_inches=0.1,
                )

                print(f"[SAVED] {output_file}")

                #plt.show()


    # -----------------------------
    # Save omitted CI records
    # -----------------------------

    if len(omitted_ci_rows) > 0:
        omitted_ci_df = pd.DataFrame(omitted_ci_rows)
        omitted_ci_file = "omitted_ci95_large_intervals.csv"

        omitted_ci_df.to_csv(omitted_ci_file, index=False)

        print(f"[SAVED] {omitted_ci_file}")
