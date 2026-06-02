from BenchmarkDPFair.Benchmark import BenchmarkDatasetConfig, BenchmarkInfo, benchmark
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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
classifiers = [lr, rf, xgb]
ckwargs = [
    ESTIMATOR_PARAMS,
    {},
    {"objective": 'binary:logistic'},
]
classifier_name = ["LR", "RF", "XGB"]

combinations = [
    (0, 0),
    (1, 0),
    (2, 0),
    (2, 1),
]

synths = ["aim", "mst"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Receive n numbers from CLI")

    parser.add_argument(
        "seeds",
        nargs="+",        # 1 or more values
        type=int          # convert automatically to int
    )

    args = parser.parse_args()
    seeds = args.seeds
    max_iter = [50, 75, 100, 125, 150]
    nu = [None, 0.01, 0.05, 0.1, 0.5]
    grid_size = [10, 15, 20, 30, 50]
    grid_limit = [2.0, 1.0, 1.5, 2.5, 3.0]

    mkwarg_pairs = [(i, j) for i in range(len(max_iter)) for j in range(len(max_iter))]

    for clf_idx, syn_idx in combinations:
        for i, j in mkwarg_pairs:
            classifier = classifiers[clf_idx]
            synth = synths[syn_idx]

            benchmark_config = BenchmarkInfo(
                dp_method=synth,
                output_dir=f"./ablation/in-processing/Compas/output/{classifier_name[clf_idx]}/",
                seeds=seeds,
                eps = [0.05, 0.1, .25, .5, .75, 1, 2, 3, 5, 10, 15, 20],
                classifier=classifier,
                classifier_kwargs=ckwargs[clf_idx],
                mitigator_kwargs={
                    'egr': {
                        'max_iter': max_iter[i],
                        'nu': nu[j],
                    },
                    'gsr': {
                        'grid_size': grid_size[i],
                        'grid_limit': grid_limit[j]
                    }
                }
            )

            benchmark_dataset = BenchmarkDatasetConfig(
                name = "Compas",
                target= "two_year_recid",
                root_dir="../data",
                sensitive_attr = "race",
                index_col="Unnamed: 0",
                categorical_cols = ['race', 'score_text', 'c_charge_degree','age', 'sex', 'two_year_recid'],
                ordinal_cols=["priors_count"],
                sensitive_cols = ['race', 'sex'],
            )


            benchmark(benchmark_info=benchmark_config, data_conf=benchmark_dataset)
            del benchmark_dataset, benchmark_config
