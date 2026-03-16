from BenchmarkDPFair.Benchmark import BenchmarkDatasetConfig, BenchmarkInfo, benchmark
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
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
classifiers = [lr, rf, xgb]
ckwargs = [
    ESTIMATOR_PARAMS,
    {},
    {"objective": 'binary:logistic'}
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


    for clf_idx, syn_idx in combinations:
        classifier = classifiers[clf_idx]
        synth = synths[syn_idx]
                
        benchmark_config = BenchmarkInfo(
            dp_method=synth,
            output_dir=f"./data/Adult/output/{classifier_name[clf_idx]}/",
            seeds = seeds,
            eps = [.25, .5, .75, 1, 5, 10, 15, 20],
            classifier=classifier,
            classifier_kwargs=ckwargs[clf_idx]
        )

        benchmark_dataset = BenchmarkDatasetConfig(
            name = "Adult",
            target= "income",
            root_dir="../data",
            sensitive_attr = "sex",
            index_col="Unnamed: 0",
            categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income', 'age', 'hours-per-week'],
            sensitive_cols = ['race', 'sex'],
        )

        benchmark(benchmark_info=benchmark_config, data_conf=benchmark_dataset)
        
