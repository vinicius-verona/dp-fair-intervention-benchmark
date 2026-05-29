import argparse
from typing import List, Union
from BenchmarkDPFair.DataGenerator import generate_data, DatasetGeneratorConfig

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
    
    # eps : List[Union[int,float]] = [.05, .1, .25, .5, .75, 1, 2, 3, 5, 10, 15, 20]
    eps : List[Union[int,float]] = [.05, .1, 2, 3]

    for synthesizer in ["aim", "mst"]:
        for s in seeds:
            data_conf = DatasetGeneratorConfig(
                name = "Compas",
                target= "two_year_recid",
                synthesizer = synthesizer,
                root_dir="../data",
                sensitive_attr = "race",
                # index_col="Unnamed: 0",
                categorical_cols = ['race', 'score_text', 'c_charge_degree','age', 'sex', 'two_year_recid'],
                sensitive_cols = ['race', 'sex'],
                ordinal_cols = ['priors_count'],
                privacy_budgets=eps,
                binary_encoder=binary_encode,
                seed = s,
                test_split_size=0.4#0.2,
                # cal_split_size=0.2
            )

            generate_data(f"compas.csv", "", data_conf, verbose=True)