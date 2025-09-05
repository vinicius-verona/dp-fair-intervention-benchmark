from BenchmarkDPFair.DataGenerator import generate_data, DatasetGeneratorConfig

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


data_conf = DatasetGeneratorConfig(
    name = "Adult",
    target= "income",
    synthesizer = "aim",
    root_dir="./data",
    sensitive_attr = "sex",
    index_col="Unnamed: 0",
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income', 'age', 'hours-per-week'],
    sensitive_cols = ['race', 'sex'],
    privacy_budgets=[.25],
    binary_encoder=binary_encode
)

generate_data("adult.csv", data_conf, verbose=True)