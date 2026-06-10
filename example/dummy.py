import argparse
from typing import List, Union
from BenchmarkDPFair.DataGenerator import generate_data, DatasetGeneratorConfig
from BenchmarkDPFair.Benchmark import benchmark, BenchmarkInfo, BenchmarkDatasetConfig
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

ESTIMATOR_PARAMS = {
    'max_iter': 10000,
    'solver': 'saga',
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
    (0, 1),
]

synths = ["aim", "mst"]

def binary_encode(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col == 'sex':
            df[col] = df[col].apply(lambda x: 1 if x == 'Male' or x == 1 else 0)
        elif col == 'race':
            df[col] = df[col].apply(lambda x: 1 if x == "Caucasian" or x == 1 else 0)
        elif col == 'age':
            df[col] = ((df[col] >= 25) & (df[col] <= 45)).astype(int)
        else:
            most_common_value = df[col].mode()[0]
            df[col] = (df[col] != most_common_value).astype(int)
    return df

def filter_compas(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df['race'] == "African-American") | (df['race'] == "Caucasian")]

    if 'days_b_screening_arrest' in df.columns:
        df = df[(df['days_b_screening_arrest'] <= 30) & (df['days_b_screening_arrest'] >= -30)]
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments of Data Generation for Adult")

    parser.add_argument(
        "--seeds", "-s",
        nargs="+",        # 1 or more values
        required=True,    
        type=int          # convert automatically to int
    )

    args = parser.parse_args()
    seeds = args.seeds
    
    eps : List[Union[int,float]] = [.05, .1]

    for synthesizer in synths:
        for s in seeds:
            data_conf = DatasetGeneratorConfig(
                name = "Compas",
                target= "two_year_recid",
                synthesizer = synthesizer,
                root_dir="./data",
                sensitive_attr = "race",
                categorical_cols = ['race', 'score_text', 'c_charge_degree','age', 'sex', 'two_year_recid'],
                sensitive_cols = ['race', 'sex'],
                ordinal_cols = ['priors_count'],
                privacy_budgets=eps,
                binary_encoder=binary_encode,
                seed = s,
                test_split_size=0.4,
                data_filter = filter_compas
            )

            generate_data(f"compas.csv", "", data_conf, "./data/Compas/", verbose=True)

    for clf_idx, syn_idx in combinations:
        classifier = classifiers[clf_idx]
        synth = synths[syn_idx]

        benchmark_config = BenchmarkInfo(
            dp_method=synth,
            output_dir=f"./output/Dummy-Compas/{classifier_name[clf_idx]}/",
            seeds=seeds,
            eps = eps,
            classifier=classifier,
            classifier_kwargs=ckwargs[clf_idx]
        )

        benchmark_dataset = BenchmarkDatasetConfig(
            name = "Compas",
            target= "two_year_recid",
            root_dir="./data",
            sensitive_attr = "race",
            index_col="Unnamed: 0",
            categorical_cols = ['race', 'score_text', 'c_charge_degree','age', 'sex', 'two_year_recid'],
            ordinal_cols=["priors_count"],
            sensitive_cols = ['race', 'sex'],
        )


        benchmark(benchmark_info=benchmark_config, data_conf=benchmark_dataset)