import numpy as np
import pandas as pd

import folktables
from folktables import ACSDataSource


# Load full dataset
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
data = data_source.get_data(states=["UT"], download=True)

ACSIncomeN = folktables.BasicProblem(
    features=[
        'AGEP',
        'COW',
        'SCHL',
        'MAR',
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
        'SEX',
        'RAC1P',
    ],
    target='PINCP',
    preprocess=folktables.adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

df_X, df_y, _ = ACSIncomeN.df_to_pandas(data)
median = df_y.median()
df_y = (df_y > median).astype(int)

print(f"Threshold for target: {median}")

# Remove null values from dataset and its respective label
null_indices = df_X[df_X.isnull().any(axis=1)].index

# Drop those indices from both X and y
df_X = df_X.drop(null_indices)
df_y = df_y.drop(null_indices)

if not df_X.index.equals(df_y.index):
    raise Exception("Different indexes between df_X and df_y.")

df = pd.concat([df_X, df_y], axis=1)
df.to_csv("acsincome.csv")