from BenchmarkDPFair.Benchmark import BenchmarkDatasetConfig, BenchmarkInfo, benchmark
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# from tabicl import TabICLClassifier
# from pytorch_tabnet.tab_model import TabNetClassifier


# seeds  of paper -> 
# [ 5,42,253,4112,32645,
#   602627,153073,53453,178753,243421,
#   767707,113647,796969,553067,96797,
#   133843,6977,460403,126613,583879 ],


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
    # (0, 0),
    # (1, 0),
    (2, 0),
    # (1, 1),
    (2, 1),
    # (3, 1),
    # (4, 1),
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
            output_dir=f"./data/ACSIncome/output/{classifier_name[clf_idx]}/",
            seeds=seeds,
            eps = [0.05, 0.1, .25, .5, .75, 1, 2, 3, 5, 10, 15, 20],
            classifier=classifier,
            classifier_kwargs=ckwargs[clf_idx]
        )

        benchmark_dataset = BenchmarkDatasetConfig(
            name = "ACSIncome",
            target= "PINCP",
            root_dir="../data",
            sensitive_attr = "SEX",
            index_col="Unnamed: 0",
            categorical_cols = ['COW','MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P','PINCP'],
            ordinal_cols=["SCHL", "AGEP"],
            sensitive_cols = ['SEX', 'RAC1P'],
        )

        benchmark(benchmark_info=benchmark_config, data_conf=benchmark_dataset)
        
