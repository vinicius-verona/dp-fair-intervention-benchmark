from BenchmarkDPFair.Benchmark import BenchmarkDatasetConfig, BenchmarkInfo, benchmark

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# from tabicl import TabICLClassifier
# from pytorch_tabnet.tab_model import TabNetClassifier

import argparse

ESTIMATOR_PARAMS = {
    'max_iter': 10000,
    'solver': 'saga',
    'penalty': 'elasticnet',
    'l1_ratio': 0.5,
    'C': 0.8
}

lr = LogisticRegression
rf = RandomForestClassifier
xgb = XGBClassifier
# tn = TabNetClassifier
# ti = TabICLClassifier
classifiers = [lr, rf, xgb]#, tn, ti]
ckwargs = [
    ESTIMATOR_PARAMS,
    {},
    {"objective": 'binary:logistic'},
    # {},
    # {}
]
classifier_name = ["LR", "RF", "XGB"]#, "TN", "TI"]

combinations = [
    # (3, 0), # TN + AIM
    # (4, 0), # TI + AIM
    (0, 0),
    (1, 0),
    (2, 0),
    # (1, 1),
    # (2, 1),
    # (3, 1),
    # (4, 1),
]

synths = ["aim", "mst"]
bod_combo = 5


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Receive n numbers from CLI")

    parser.add_argument(
        "seeds",
        nargs="+",        # 1 or more values
        type=int          # convert automatically to int
    )
    parser.add_argument(
        "--combo",
        type=int,
        default=5,
        help="Combo value (default: 5, options: 1 to 6)"
    )

    args = parser.parse_args()

    seeds = args.seeds
    bod_combo = args.combo

    if bod_combo == 5:
        combinations = [
            # (3, 0), # TN + AIM
            # (4, 0), # TI + AIM
            # (0, 0),
            # (1, 0),
            # (2, 0),
            # (1, 1),
            (2, 1),
            # (3, 1),
            # (4, 1),
        ]

    for clf_idx, syn_idx in combinations:
        classifier = classifiers[clf_idx]
        synth = synths[syn_idx]

        benchmark_config = BenchmarkInfo(
            dp_method=synth,
            output_dir=f"./data/BoD/BoD-{bod_combo}/output/{classifier_name[clf_idx]}/",
            seeds=seeds,
            eps = [0.05, 0.1, .25, .5, .75, 1, 2, 3, 5, 10, 15, 20],
            classifier=classifier,
            classifier_kwargs=ckwargs[clf_idx]
        )

        benchmark_dataset = BenchmarkDatasetConfig(
            name = f"BoD-{bod_combo}",
            target= "Y",
            root_dir="../data/BoD",
            sensitive_attr = "A",
            index_col="Unnamed: 0",
            categorical_cols = ['Q', 'A', 'Y'],
            ordinal_cols=[],
            continuous_cols=['R']#['P', 'R']
        )


        benchmark(benchmark_info=benchmark_config, data_conf=benchmark_dataset)
        
