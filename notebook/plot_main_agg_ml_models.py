





import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

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

ML_MODELS = ['LR', 'RF', 'XGB']
UTILITY_METRICS = ['ACC']#, 'F1']
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
    'reweigh-XGB': 'RW-XGB',
    'reweigh-RF': 'RW-RF',
    'reweigh-LR': 'RW-LR',
    'egr-XGB': 'EGR-XGB',
    'egr-RF': 'EGR-RF',
    'egr-LR': 'EGR-LR',
    'roc-XGB': 'ROC-XGB',
    'roc-RF': 'ROC-RF',
    'roc-LR': 'ROC-LR',
}

# DATASETS = ['adult', 'folktables', 'compas', 'bod']
DATASETS = ['Adult', 'ACSIncome', 'Compas', 'BoD']
AVERAGE_BY_SEED = True

DP_ONLY_LABEL   = 'DP-ONLY'
FAIR_ONLY_LABEL = r'FAIR-ONLY ($\varepsilon=\infty$)'

# =========================
# Categorization
# =========================
intervention_map = {
    'PRE': ['reweigh'],
    'IN': ['egr'],
    'POST': ['roc']
}

label_intervention_model_map = {
    'PRE': ['reweigh-XGB', 'reweigh-RF', 'reweigh-LR'],
    'IN': ['egr-XGB', 'egr-RF', 'egr-LR'],
    'POST': ['roc-XGB', 'roc-RF', 'roc-LR']
}

method_to_cat = {m: cat for cat, ms in intervention_map.items() for m in ms}
method_to_cat[DP_ONLY_LABEL] = 'BASE'

if INTERVENTION.lower() == 'all':
    fair_methods = label_intervention_model_map['PRE'] + label_intervention_model_map['IN'] + label_intervention_model_map['POST']
elif INTERVENTION.upper() in label_intervention_model_map:
    fair_methods = label_intervention_model_map[INTERVENTION.upper()]
else:
    raise ValueError("INTERVENTION must be one of: 'all', 'PRE', 'IN', 'POST'.")

all_dp_only = [DP_ONLY_LABEL + '-XGB', DP_ONLY_LABEL + '-RF', DP_ONLY_LABEL + '-LR']
all_methods = fair_methods + all_dp_only

# =========================
# Visual mapping
# =========================
universe_methods = label_intervention_model_map['PRE'] + label_intervention_model_map['IN'] + label_intervention_model_map['POST'] + all_dp_only
colors_map = {m: plt.cm.tab20(i % 20) for i, m in enumerate(universe_methods)}

markers_map = {
    all_dp_only[0]: 'X',
    all_dp_only[1]: 'X',
    all_dp_only[2]: 'X',
    'reweigh-XGB': 'o',
    'reweigh-RF': 'o',
    'reweigh-LR': 'o',
    'egr-XGB': 's',
    'egr-RF': 's',
    'egr-LR': 's',
    'roc-XGB': '^',
    'roc-RF': '^',
    'roc-LR': '^',
}

# =========================
# Intervention stage mapping
# =========================
method_to_stage = {
    m: stage for stage, methods in intervention_map.items() for m in methods
}

STAGE_ORDER = ['PRE', 'IN', 'POST']

STAGE_COLORS = {
    'DP-ONLY': 'magenta',
    'PRE': '#1f77b4', #1f35b4 561fb4
    'IN': '#ff7f0e', #ff460e  a14040
    'POST': '#2ca02c', #7ba02c a0962c
}

STAGE_MARKERS = {
    'DP-ONLY': 'X',
    'PRE': 'o',
    'IN': 's',
    'POST': '^',
}

ML_STAGE_MARKERS = {
    'DP-ONLY-XGB': 'X',
    'DP-ONLY-RF': 'X',
    'DP-ONLY-LR': 'X',
    'PRE-XGB': 'o',
    'PRE-RF': 'o',
    'PRE-LR': 'o',
    'IN-XGB': 's',
    'IN-RF': 's',
    'IN-LR': 's',
    'POST-XGB': '^',
    'POST-RF': '^',
    'POST-LR': '^'
}



# =========================
# Processing and plotting
# =========================
#
# New expected structure:
#
# root/
#   ACSIncome/
#     output/<ML_MODEL>/ACSIncome/<DP_SYNTHETIZER>/results/*.csv
#
#   Adult/
#     output/<ML_MODEL>/Adult/<DP_SYNTHETIZER>/results/*.csv
#
#   Compas/
#     output/<ML_MODEL>/Compas/<DP_SYNTHETIZER>/results/*.csv
#
#   BoD/
#     BoD-1/
#       output/<ML_MODEL>/BoD-1/<DP_SYNTHETIZER>/results/*.csv
#     ...
#     BoD-6/
#       output/<ML_MODEL>/BoD-6/<DP_SYNTHETIZER>/results/*.csv
#
# Example:
# root/BoD/BoD-5/output/XGB/BoD-5/aim/results/*.csv
# =========================


def _normalize_dataset_name(dataset):
    """
    Normalize dataset names to match folder/file conventions.
    """
    dataset_str = str(dataset)
    lower = dataset_str.lower()

    if lower.startswith("bod-"):
        try:
            combo = int(lower.split("-", 1)[1])
            return f"BoD-{combo}"
        except Exception:
            return dataset_str

    aliases = {
        "adult": "Adult",
        "folktables": "ACSIncome",
        "acs": "ACSIncome",
        "acs_income": "ACSIncome",
        "acs-income": "ACSIncome",
        "acsincome": "ACSIncome",
        "compas": "Compas",
        "bod": "BoD",
        "biasondemand": "BoD",
        "bias_on_demand": "BoD",
        "bias-on-demand": "BoD",
    }

    return aliases.get(lower, dataset_str)


def _is_bod_dataset(dataset):
    """
    True if dataset corresponds to Bias on Demand / BoD.
    """
    normalized = _normalize_dataset_name(dataset)
    return normalized == "BoD" or str(normalized).lower().startswith("bod-")


def _bod_combo_from_dataset(dataset, default_combo=None):
    """
    Extract BoD combo from BoD-i names.

    If dataset is simply BoD, use default_combo.
    """
    normalized = _normalize_dataset_name(dataset)

    if str(normalized).lower().startswith("bod-"):
        return int(str(normalized).split("-", 1)[1])

    if normalized == "BoD":
        return default_combo

    return None


def _dataset_folder_candidates(dataset, combo=None):
    """
    Possible names for each dataset in folders/files.
    """
    normalized = _normalize_dataset_name(dataset)

    if normalized == "Adult":
        return ["Adult", "ADULT", "adult"]

    if normalized == "ACSIncome":
        return [
            "ACSIncome",
            "ACSincome",
            "acsincome",
            "ACS_Income",
            "acs_income",
            "folktables",
            "Folktables",
        ]

    if normalized == "Compas":
        return ["Compas", "COMPAS", "compas"]

    if _is_bod_dataset(normalized):
        bod_combo = _bod_combo_from_dataset(normalized, default_combo=combo)

        if bod_combo is not None:
            return [
                f"BoD-{bod_combo}",
                f"BOD-{bod_combo}",
                f"bod-{bod_combo}",
                f"BoD_{bod_combo}",
                f"BOD_{bod_combo}",
                f"bod_{bod_combo}",
                f"BiasOnDemand-{bod_combo}",
                f"biasondemand-{bod_combo}",
                f"Bias_On_Demand-{bod_combo}",
            ]

        return [
            "BoD",
            "BOD",
            "bod",
            "BiasOnDemand",
            "biasondemand",
            "Bias_On_Demand",
        ]

    return [normalized, str(dataset)]


def _find_case_insensitive_dir(parent_dir, candidates):
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


def _extract_seed_from_filename(file_path):
    """
    Extract seed information from filenames such as:

        benchmark_results_seeds_5_eps_..._synth_aim.csv

    or grouped files such as:

        benchmark_results_seeds_5_42_253_4112_eps_..._synth_aim.csv
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


def _resolve_results_root():
    """
    Resolve the root directory containing dataset folders.

    Expected layout:

        root/
            ACSIncome/
            Adult/
            Compas/
            BoD/

    If automatic detection fails, define:

        RESULTS_BASE_DIR = r"C:\\path\\to\\root"
    """
    top_level_dataset_names = [
        "ACSIncome",
        "Adult",
        "Compas",
        "BoD",
    ]

    candidates = []

    if "RESULTS_BASE_DIR" in globals():
        user_path = os.path.abspath(os.path.expanduser(str(RESULTS_BASE_DIR)))

        candidates.append(user_path)

        # If user passed root/ACSIncome, root/Adult, root/Compas, or root/BoD.
        if os.path.basename(user_path).lower() in [
            name.lower() for name in top_level_dataset_names
        ]:
            candidates.append(os.path.dirname(user_path))

        # If user passed root/BoD/BoD-i.
        if os.path.basename(user_path).lower().startswith("bod-"):
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
        "Could not find ACSIncome, Adult, Compas, or BoD folder.\n\n"
        f"Current working directory:\n{cwd}\n\n"
        "Expected structure:\n"
        "root/ACSIncome/output/<ML_MODEL>/ACSIncome/<DP_SYNTHETIZER>/results/\n"
        "root/Adult/output/<ML_MODEL>/Adult/<DP_SYNTHETIZER>/results/\n"
        "root/Compas/output/<ML_MODEL>/Compas/<DP_SYNTHETIZER>/results/\n"
        "root/BoD/BoD-5/output/<ML_MODEL>/BoD-5/<DP_SYNTHETIZER>/results/\n\n"
        "Define RESULTS_BASE_DIR manually, for example:\n"
        "RESULTS_BASE_DIR = r'C:\\path\\to\\DP-Benchmark'"
    )


def _path_as_search_text(path):
    """
    Convert path to lowercase searchable text.
    """
    return os.path.normpath(str(path)).replace("\\", "/").lower()


def _path_parts_lower(path):
    """
    Return lowercase path components.
    """
    return [
        p.lower()
        for p in os.path.normpath(str(path)).replace("\\", "/").split("/")
        if p
    ]


def _contains_any_candidate(text, candidates):
    """
    True if any candidate appears in text.
    """
    text = str(text).lower()
    return any(str(c).lower() in text for c in candidates)


def _has_model_in_path_or_file(path, file_name, ML_MODEL):
    """
    Check whether ML_MODEL appears in the path or filename.
    """
    model_lower = str(ML_MODEL).lower()
    file_lower = str(file_name).lower()
    parts_lower = _path_parts_lower(path)

    if model_lower in parts_lower:
        return True

    if file_lower.startswith(f"{model_lower}_"):
        return True

    if f"_{model_lower}_" in file_lower:
        return True

    if f"/{model_lower}/" in _path_as_search_text(path):
        return True

    return False


def _has_synth_in_path_or_file(path, file_name, DP_SYNTHETIZER):
    """
    Check whether synthesizer appears in path or filename.
    """
    synth_lower = str(DP_SYNTHETIZER).lower()
    file_lower = str(file_name).lower()
    parts_lower = _path_parts_lower(path)

    if synth_lower in parts_lower:
        return True

    if f"_synth_{synth_lower}" in file_lower:
        return True

    if f"_{synth_lower}_" in file_lower:
        return True

    if file_lower.endswith(f"_{synth_lower}.csv"):
        return True

    return False


def _has_dataset_in_path_or_file(path, file_name, dataset, combo=None):
    """
    Check whether dataset appears in path or filename.

    For BoD, when combo is provided, it must match BoD-<combo>.
    """
    candidates = _dataset_folder_candidates(dataset, combo=combo)
    text = _path_as_search_text(path) + "/" + str(file_name).lower()

    return _contains_any_candidate(text, candidates)


def _is_csv_candidate(file_name):
    """
    Keep only benchmark CSV files.
    """
    file_lower = str(file_name).lower()

    if not file_lower.endswith(".csv"):
        return False

    if file_lower.startswith("benchmark_results_seeds_"):
        return True

    if "_results_" in file_lower:
        return True

    return False


def _get_results_dir(
    results_root,
    ML_MODEL,
    dataset,
    DP_SYNTHETIZER,
    combo=None,
):
    """
    Resolve the exact results directory when using the deep benchmark structure.

    Regular datasets:
        root/<DATASET>/output/<ML_MODEL>/<DATASET>/<SYNTH>/results/

    BoD:
        root/BoD/BoD-i/output/<ML_MODEL>/BoD-i/<SYNTH>/results/
    """
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

    if _is_bod_dataset(normalized_dataset):
        bod_combo = _bod_combo_from_dataset(
            normalized_dataset,
            default_combo=combo,
        )

        if bod_combo is None:
            return None

        bod_root = _find_case_insensitive_dir(
            results_root,
            ["BoD", "BOD", "bod"],
        )
        if bod_root is None:
            return None

        bod_combo_dir = _find_case_insensitive_dir(
            bod_root,
            _dataset_folder_candidates("BoD", combo=bod_combo),
        )
        if bod_combo_dir is None:
            return None

        output_dir = _find_case_insensitive_dir(
            bod_combo_dir,
            ["output", "Output"],
        )
        if output_dir is None:
            return None

        model_dir = _find_case_insensitive_dir(
            output_dir,
            model_candidates,
        )
        if model_dir is None:
            return None

        dataset_dir_2 = _find_case_insensitive_dir(
            model_dir,
            _dataset_folder_candidates("BoD", combo=bod_combo),
        )
        if dataset_dir_2 is None:
            return None

        synth_dir = _find_case_insensitive_dir(
            dataset_dir_2,
            synth_candidates,
        )
        if synth_dir is None:
            return None

        results_dir = _find_case_insensitive_dir(
            synth_dir,
            ["results", "Results"],
        )

        return results_dir

    dataset_candidates = _dataset_folder_candidates(normalized_dataset)

    dataset_dir_1 = _find_case_insensitive_dir(
        results_root,
        dataset_candidates,
    )
    if dataset_dir_1 is None:
        return None

    output_dir = _find_case_insensitive_dir(
        dataset_dir_1,
        ["output", "Output"],
    )
    if output_dir is None:
        return None

    model_dir = _find_case_insensitive_dir(
        output_dir,
        model_candidates,
    )
    if model_dir is None:
        return None

    dataset_dir_2 = _find_case_insensitive_dir(
        model_dir,
        dataset_candidates,
    )
    if dataset_dir_2 is None:
        return None

    synth_dir = _find_case_insensitive_dir(
        dataset_dir_2,
        synth_candidates,
    )
    if synth_dir is None:
        return None

    results_dir = _find_case_insensitive_dir(
        synth_dir,
        ["results", "Results"],
    )

    return results_dir


def _find_seed_csvs(
    results_root,
    ML_MODEL,
    dataset,
    DP_SYNTHETIZER,
    combo=None,
):
    """
    Find CSV files under the new root-based structure.

    It first tries the exact deep path. If not found, it falls back to
    walking the relevant dataset folder.
    """
    results_dir = _get_results_dir(
        results_root=results_root,
        ML_MODEL=ML_MODEL,
        dataset=dataset,
        DP_SYNTHETIZER=DP_SYNTHETIZER,
        combo=combo,
    )

    csv_files = []

    if results_dir is not None and os.path.isdir(results_dir):
        for file_name in os.listdir(results_dir):
            if not _is_csv_candidate(file_name):
                continue

            file_path = os.path.join(results_dir, file_name)

            if not os.path.isfile(file_path):
                continue

            if not _has_model_in_path_or_file(file_path, file_name, ML_MODEL):
                continue

            if not _has_synth_in_path_or_file(file_path, file_name, DP_SYNTHETIZER):
                continue

            csv_files.append(file_path)

        csv_files = sorted(set(csv_files), key=lambda p: os.path.basename(p))

        return csv_files, results_dir

    # Fallback recursive search.
    normalized_dataset = _normalize_dataset_name(dataset)

    if _is_bod_dataset(normalized_dataset):
        bod_combo = _bod_combo_from_dataset(
            normalized_dataset,
            default_combo=combo,
        )

        bod_root = _find_case_insensitive_dir(
            results_root,
            ["BoD", "BOD", "bod"],
        )

        if bod_root is None:
            searched_root = results_root
        else:
            searched_root = _find_case_insensitive_dir(
                bod_root,
                _dataset_folder_candidates("BoD", combo=bod_combo),
            )

            if searched_root is None:
                searched_root = bod_root

    else:
        searched_root = _find_case_insensitive_dir(
            results_root,
            _dataset_folder_candidates(normalized_dataset),
        )

        if searched_root is None:
            searched_root = results_root

    if not os.path.isdir(searched_root):
        return [], searched_root

    for dirpath, dirnames, filenames in os.walk(searched_root):
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in {
                ".git",
                "__pycache__",
                ".ipynb_checkpoints",
                "log",
                "logs",
            }
        ]

        for file_name in filenames:
            if not _is_csv_candidate(file_name):
                continue

            file_path = os.path.join(dirpath, file_name)

            if not os.path.isfile(file_path):
                continue

            if not _has_model_in_path_or_file(file_path, file_name, ML_MODEL):
                continue

            if not _has_synth_in_path_or_file(file_path, file_name, DP_SYNTHETIZER):
                continue

            if not _has_dataset_in_path_or_file(
                file_path,
                file_name,
                dataset,
                combo=combo,
            ):
                continue

            csv_files.append(file_path)

    csv_files = sorted(set(csv_files), key=lambda p: os.path.basename(p))

    return csv_files, searched_root


def _read_seed_csvs(
    results_root,
    ML_MODEL,
    dataset,
    DP_SYNTHETIZER,
    combo=None,
):
    """
    Read and concatenate CSV files from the new root-based structure.

    For BoD, combo is used to locate/read:
        BoD-1, BoD-2, ..., BoD-6.
    """
    csv_files, searched_root = _find_seed_csvs(
        results_root=results_root,
        ML_MODEL=ML_MODEL,
        dataset=dataset,
        DP_SYNTHETIZER=DP_SYNTHETIZER,
        combo=combo,
    )

    if len(csv_files) == 0:
        print(
            f"[SKIP] No CSV files found for "
            f"ML_MODEL={ML_MODEL}, dataset={dataset}, "
            f"normalized_dataset={_normalize_dataset_name(dataset)}, "
            f"combo={combo}, "
            f"synth={DP_SYNTHETIZER}, "
            f"RESULTS_ROOT={results_root}, "
            f"searched_root={searched_root}"
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

        if _is_bod_dataset(dataset):
            tmp["BoD-Combo"] = combo

        dfs.append(tmp)

    df = pd.concat(dfs, ignore_index=True)

    if _is_bod_dataset(dataset) and combo is not None:
        if "Combo" in df.columns:
            df = df[df["Combo"].astype(str) == str(combo)].copy()
        elif "COMBO" in df.columns:
            df = df[df["COMBO"].astype(str) == str(combo)].copy()
        elif "combo" in df.columns:
            df = df[df["combo"].astype(str) == str(combo)].copy()
        elif "BoD-Combo" in df.columns:
            df = df[df["BoD-Combo"].astype(str) == str(combo)].copy()

    if df.empty:
        print(
            f"[SKIP] Empty dataframe after reading/filtering for "
            f"ML_MODEL={ML_MODEL}, dataset={dataset}, "
            f"combo={combo}, synth={DP_SYNTHETIZER}."
        )
        return None

    print(
        f"[READ] "
        f"{ML_MODEL} | dataset={dataset} -> {_normalize_dataset_name(dataset)} | "
        f"combo={combo} | {DP_SYNTHETIZER}: {len(csv_files)} CSV files, {len(df)} rows"
    )

    for csv_file in csv_files:
        print(f"[FILE] {csv_file}")

    return df


RESULTS_ROOT = _resolve_results_root()

print(f"[RESULTS_ROOT] {RESULTS_ROOT}")
print(f"[CWD] {os.getcwd()}")


for UTILITY_METRIC in UTILITY_METRICS:
    for COMBO in [1, 2, 3, 4, 5, 6]:
        for dataset in DATASETS:

            normalized_dataset = _normalize_dataset_name(dataset)

            if not _is_bod_dataset(normalized_dataset) and COMBO != 1:
                continue

            if _is_bod_dataset(normalized_dataset):
                if str(normalized_dataset).lower().startswith("bod-"):
                    current_combo = _bod_combo_from_dataset(normalized_dataset)

                    if COMBO != current_combo:
                        continue
                else:
                    current_combo = COMBO
            else:
                current_combo = None

            dfs = []

            for ML_MODEL in ML_MODELS:

                df = _read_seed_csvs(
                    results_root=RESULTS_ROOT,
                    ML_MODEL=ML_MODEL,
                    dataset=normalized_dataset,
                    DP_SYNTHETIZER=DP_SYNTHETIZER,
                    combo=current_combo,
                )

                if df is None:
                    continue

                df = df.replace([np.inf, -np.inf], np.nan)

                # ---- Compute F1-score if requested ----
                if UTILITY_METRIC == "F1":
                    if not {"PREC", "REC"}.issubset(df.columns):
                        raise ValueError("F1 requires PREC and REC columns in the CSV.")

                    df["F1"] = np.where(
                        (df["PREC"] + df["REC"]) > 0,
                        2 * df["PREC"] * df["REC"] / (df["PREC"] + df["REC"]),
                        0.0,
                    )

                dp_only_df = df[
                    df["Fair-Method"].isna()
                    & df["Epsilon"].notna()
                ].copy()

                dp_only_df["Fair-Method"] = DP_ONLY_LABEL + "-" + ML_MODEL

                fair_df = df[
                    df["Fair-Method"].isin(
                        intervention_map["PRE"]
                        + intervention_map["IN"]
                        + intervention_map["POST"]
                    )
                ].copy()

                fair_df["Fair-Method"] = fair_df["Fair-Method"] + "-" + ML_MODEL

                combined_df = pd.concat([fair_df, dp_only_df], ignore_index=True)

                if AVERAGE_BY_SEED:
                    combined_df = (
                        combined_df
                        .groupby(["Fair-Method", "Epsilon"], dropna=False)
                        .mean(numeric_only=True)
                        .reset_index()
                    )

                if not combined_df.empty:
                    dfs.append(combined_df)

            # =========================
            # Plot
            # =========================
            if len(dfs) == 0:
                print(
                    f"[SKIP] Nothing to plot for "
                    f"dataset={normalized_dataset}, combo={current_combo}, "
                    f"UTILITY_METRIC={UTILITY_METRIC}, synth={DP_SYNTHETIZER}."
                )
                continue

            fig, axes = plt.subplots(1, 3, figsize=(22, 5.1))
            df_dataset = pd.concat(dfs, ignore_index=True)

            for c, FAIRNESS_METRIC in enumerate(FAIRNESS_METRICS):
                ax = axes[c]

                for method in all_methods:
                    data = df_dataset[df_dataset["Fair-Method"] == method]

                    if data.empty:
                        continue

                    data = data.sort_values(
                        "Epsilon",
                        key=lambda x: x.fillna(999),
                    )

                    if len(data) > 1:
                        ax.plot(
                            data[FAIRNESS_METRIC],
                            data[UTILITY_METRIC],
                            linestyle="--",
                            linewidth=1.2,
                            color=colors_map[method],
                            alpha=0.8,
                        )

                    for _, row in data.iterrows():
                        fair_method_ml = row["Fair-Method"]
                        eps = row["Epsilon"]

                        size = (
                            320
                            if pd.isna(eps)
                            else 100 + (min(eps, 20) / 20) * 200
                        )

                        if pd.isna(eps):
                            ax.scatter(
                                row[FAIRNESS_METRIC],
                                row[UTILITY_METRIC],
                                marker=markers_map[fair_method_ml],
                                s=size,
                                facecolors="white",
                                edgecolors=colors_map[fair_method_ml],
                                linewidths=2.5,
                                zorder=10,
                            )
                        else:
                            ax.scatter(
                                row[FAIRNESS_METRIC],
                                row[UTILITY_METRIC],
                                marker=markers_map[fair_method_ml],
                                s=size,
                                color=colors_map[fair_method_ml],
                                edgecolors="k",
                                alpha=1,
                            )

                ax.set_xlabel(METRIC_LABELS[FAIRNESS_METRIC])

                if c == 0:
                    ax.set_ylabel(METRIC_LABELS[UTILITY_METRIC])

                ax.grid(True, linestyle=":", alpha=0.6)

            # =========================
            # Legend
            # =========================
            legend_items = [
                plt.Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markeredgewidth=2.5,
                    markersize=14,
                    label=FAIR_ONLY_LABEL,
                )
            ]

            legend_items += [
                plt.Line2D([0], [0], linestyle="none", label=""),
                plt.Line2D([0], [0], linestyle="none", label=""),
            ]

            for m in all_dp_only:
                legend_items.append(
                    plt.Line2D(
                        [0], [0],
                        marker="X",
                        linestyle="none",
                        markerfacecolor=colors_map[m],
                        markeredgecolor="k",
                        markersize=14,
                        label=m,
                    )
                )

            if INTERVENTION.lower() in ["all", "pre"]:
                for m in label_intervention_model_map["PRE"]:
                    legend_items.append(
                        plt.Line2D(
                            [0], [0],
                            marker=markers_map[m],
                            linestyle="--",
                            color=colors_map[m],
                            markeredgecolor="k",
                            markersize=10,
                            label=f"PRE: {METHOD_ACRONYMS[m]}",
                        )
                    )

            if INTERVENTION.lower() in ["all", "in"]:
                for m in label_intervention_model_map["IN"]:
                    legend_items.append(
                        plt.Line2D(
                            [0], [0],
                            marker=markers_map[m],
                            linestyle="--",
                            color=colors_map[m],
                            markeredgecolor="k",
                            markersize=10,
                            label=f"IN: {METHOD_ACRONYMS[m]}",
                        )
                    )

            if INTERVENTION.lower() in ["all", "post"]:
                for m in label_intervention_model_map["POST"]:
                    legend_items.append(
                        plt.Line2D(
                            [0], [0],
                            marker=markers_map[m],
                            linestyle="--",
                            color=colors_map[m],
                            markeredgecolor="k",
                            markersize=10,
                            label=f"POST: {METHOD_ACRONYMS[m]}",
                        )
                    )

            fig.legend(
                handles=legend_items,
                loc="upper center",
                ncol=5 if INTERVENTION.lower() == "all" else 2,
                bbox_to_anchor=(0.5, 1.13),
            )

            plt.tight_layout(rect=[0, 0, 1, 0.92])

            output_dataset_name = _normalize_dataset_name(normalized_dataset)

            if _is_bod_dataset(output_dataset_name):
                figname = (
                    f"fig_results_all_models_"
                    f"{DP_SYNTHETIZER}_{UTILITY_METRIC}_{INTERVENTION}_"
                    f"BoD-{current_combo}.pdf"
                )
            else:
                figname = (
                    f"fig_results_all_models_"
                    f"{DP_SYNTHETIZER}_{UTILITY_METRIC}_{INTERVENTION}_"
                    f"{output_dataset_name}.pdf"
                )

            plt.savefig(
                figname,
                dpi=500,
                bbox_inches="tight",
                pad_inches=0.1,
            )

            print(f"[SAVED] {figname}")

            plt.show()
