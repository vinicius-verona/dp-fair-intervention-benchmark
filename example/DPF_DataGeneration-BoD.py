import argparse
from typing import List, Union
from BenchmarkDPFair.DataGenerator import generate_data, DatasetGeneratorConfig
import pandas as pd

sensitive_columns = ['A']
sensitive_attr = 'A'

# Non-sensitive columns
non_sensitive_columns = ['Q', 'R']

# Target column
target_column = 'Y'

categorical_columns = ['Q', 'A', 'Y']
ordinal_columns = []
continuous_columns = ['R']


def binary_encode(df, columns):
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


def compress_data(df):
    return df

def discretize_features(df, features_to_discretize, num_bins=5):
    for col in features_to_discretize:
        if col in df.columns:
            df[col] = pd.cut(df[col], bins=num_bins, labels=False, include_lowest=True)
            df[col] = df[col].astype(int)
    return df

def pre_process_dataset(X, y):
    ds = pd.concat([X, y], axis=1)

    global non_sensitive_columns, continuous_columns

    non_sensitive_columns = [col for col in non_sensitive_columns if col in ds.columns]
    continuous_columns = [col for col in continuous_columns if col in ds.columns]

    ds = discretize_features(ds, continuous_columns, num_bins=5)
    
    # Reverse A = 0 with A = 1, to match our sensitive attribute pattern
    if sensitive_attr == 'A':
        ds[sensitive_attr] = ds[sensitive_attr].apply(lambda x: 1 if x == 0 else 0)

    ds.index.name = None
    return ds

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
    
    eps : List[Union[int,float]] = [.05, .1, .25, .5, .75, 1, 2, 3, 5, 10, 15, 20]
    for synthesizer in ["aim", "mst"]:
        for bod in range(1,7):
            for s in seeds:
                data_conf = DatasetGeneratorConfig(
                    name = f"BoD-{bod}",
                    target= "Y",
                    synthesizer = synthesizer,
                    root_dir=f"./data/BoD/{synthesizer}/",
                    sensitive_attr = "A",
                    index_col="Unnamed: 0",
                    categorical_cols = ['Q', 'A', 'Y'],
                    sensitive_cols = ['A'],
                    ordinal_cols = [],
                    continuous_cols = ['R'],
                    privacy_budgets=eps,
                    binary_encoder=binary_encode,
                    compressor = compress_data,
                    pre_processer=pre_process_dataset,
                    seed = s,
                    test_split_size=0.4
                )

                generate_data(f"train.csv", f"test.csv", data_conf, verbose=True)