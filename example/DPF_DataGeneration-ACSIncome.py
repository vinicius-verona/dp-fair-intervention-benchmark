import argparse
from typing import List, Union
import pandas as pd

from BenchmarkDPFair.DataGenerator import generate_data, DatasetGeneratorConfig
from utils.Groups import ACSIncome_categories_group



# seeds = [ 
#     5,42,253,4112,32645,
#     602627,153073,53453,178753,243421,
#     767707,113647,796969,553067,96797,
#     133843,6977,460403,126613,583879 
# ]

def compress_dataset(df):
    categorical_columns = ['COW','MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P','PINCP']
    for col in categorical_columns:
        if col in df.columns and col == 'OCCP':
            # Compress age into bins
            df[col] = pd.cut(
                df[col],
                bins=[v[0] for v in ACSIncome_categories_group[col].values()] +
                    [list(ACSIncome_categories_group[col].values())[-1][1] + 1],
                labels=list(ACSIncome_categories_group[col].keys())
            )
                        
    return df


"""---
# **Data Preprocessing - Cleaning / Encoding**
"""
# Function to apply binary encoding
def binary_encode(df, columns):
    for col in columns:
        if col == 'SEX':
            df[col] = df[col].apply(lambda x: 1 if x == 'Male' or int(x) == 1 else 0)
        elif col == 'RAC1P':
            df[col] = df[col].apply(lambda x: 1 if x == "White alone" or int(x) == 1 else 0)
    return df

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
                name = "ACSIncome",
                target= "PINCP",
                synthesizer = synthesizer,
                root_dir="../data",
                sensitive_attr = "SEX",
                categorical_cols = ['COW','MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P','PINCP'],
                sensitive_cols = ['SEX', 'RAC1P'],
                ordinal_cols = ['SCHL', 'AGEP'],
                privacy_budgets=eps,
                binary_encoder=binary_encode,
                compressor=compress_dataset,
                seed = s,
                test_split_size=0.4
            )

            generate_data(f"acsincome.csv", "", data_conf, verbose=True)