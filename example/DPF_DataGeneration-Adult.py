import argparse
from typing import List, Union
import pandas as pd

from BenchmarkDPFair.DataGenerator import generate_data, DatasetGeneratorConfig
from utils.CountryContinentMap import country_continent_map

"""
Compress the dataset in order to reduce large-cardinality categorical columns
"""
def compress_dataset(df):
    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 
                           'relationship', 'race', 'sex', 'native-country', 'income', 'age', 'hours-per-week']
    
    for col in categorical_columns:
        if col in df.columns and col == 'age':
            # Compress age into bins
            df[col] = pd.cut(df[col], bins=[i for i in range(0, 101, 5)], labels=[i for i in range(0, 20)], right=False)
            
        elif col in df.columns and col == 'native-country':
            # Compress native-country into continent categories
            df[col] = df[col].map(country_continent_map).fillna('Other')
            
        elif col in df.columns and col == 'hours-per-week':
            # Compress age into bins
            df[col] = pd.cut(df[col], bins=[0, 20, 40, 60, 80, 100], labels=[i for i in range(0, 5)], right=False)
            
    return df


"""---
# **Data Preprocessing - Cleaning / Encoding**
"""
def binary_encode(df, columns):
    for col in columns:
        if col == 'sex':
            df[col] = df[col].apply(lambda x: 1 if x == 'Male' or x == 1 else 0)
        elif col == 'race':
            df[col] = df[col].apply(lambda x: 1 if x == 'White' or x == 4 else 0)
        else:
            most_common_value = df[col].mode()[0]
            df[col] = (df[col] != most_common_value).astype("int64")
    return df


# seeds = [ 
#     5,42,253,4112,32645,
#     602627,153073,53453,178753,243421,
#     767707,113647,796969,553067,96797,
#     133843,6977,460403,126613,583879 
# ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments of Data Generation for Adult")

    parser.add_argument(
        "--seeds", "-s",
        nargs="+",        # 1 or more values
        type=int          # convert automatically to int
    )

    args = parser.parse_args()

    seeds = args.seeds
    eps : List[Union[int,float]] = [.05, .1, .25, .5, .75, 1, 2, 3, 5, 10, 15, 20]
    
    for synthesizer in ["aim", "mst"]:
        for s in seeds:
            data_conf = DatasetGeneratorConfig(
                name = "Adult",
                target= "income",
                synthesizer = synthesizer,
                root_dir="../data",
                sensitive_attr = "sex",
                categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income', 'age', 'hours-per-week'],
                sensitive_cols = ['race', 'sex'],
                privacy_budgets=eps,
                binary_encoder=binary_encode,
                compressor=compress_dataset,
                seed = s,
                test_split_size=0.4
            )

            generate_data(f"adult.csv", "", data_conf, verbose=True)