# -*- coding: utf-8 -*-
"""
Generate-FolkTables-DP-Seeds-Epsilons
- Vinícius Gabriel Angelozzi Verona de Resende (vinicius-gabriel.angelozzi-verona-de-resende@inria.fr)
"""

import sys
import random
import folktables
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

# EPSILON = [0.25, 0.5, 0.75, 1, 5, 10, 15, 20]
EPSILON = [15, 20]
NORMALIZE = True
path = ''
DP_ALG = 'dpctgan' #'aim'

# Sensitive columns
sensitive_columns = ['SEX', 'RAC1P']
sensitive_attr = 'SEX'
ordinal_columns = ['SCHL', 'AGEP']
continuous_columns = []

# Non-sensitive columns
non_sensitive_columns = ['AGEP','COW','SCHL','MAR','OCCP','POBP','RELP','WKHP']

# Target column
target_column = 'PINCP'
categorical_columns = ['COW','MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P','PINCP']
USED_FEATURES = ['AGEP','COW','MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P', 'PINCP']


"""
Compress the dataset in order to reduce large-cardinality categorical columns
"""
def compress_dataset(df):
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

def pre_process_dataset(X, y):
    ds = pd.concat([X, y], axis=1)
    ds = compress_dataset(ds)    

    for col in categorical_columns:
        if not pd.api.types.is_numeric_dtype(ds[col]):
            ds[col] = ds[col].astype('category').cat.codes # Int encode

    # Binary encoding of sensitive columns, where 0 is the value in the category with the highest value
    ds = binary_encode(ds, sensitive_columns)#[sensitive_attr])
    
    return ds


"""DP-Using-Adult.ipynb

# Installation of Smartnoise Synth library
* [Smartnoise Synth](https://github.com/opendp/smartnoise-sdk/tree/main/synth)
"""
from snsynth import Synthesizer
from folktables import ACSDataSource, ACSIncome, generate_categories
from Groups import ACSIncome_categories_group
import os

SEED = None
if len(sys.argv) != 3:
    print("Syntax: dp_dataset_generator.py seed storage-path")
    os.abort()
    
SEED = int(sys.argv[1])
SAVE_PATH = sys.argv[2] + DP_ALG + "/"

print(f"Running dataset generator with seed {SEED}")
if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)

    # Load full dataset
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    data = data_source.get_data(states=["UT"])

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

    # Preprocess the dataset
    ds          = pre_process_dataset(df_X, df_y)
    train, test = train_test_split(ds, test_size=0.4, random_state=SEED)
    all_columns = ds.columns.tolist()

    # Ensure all columns are accounted for
    specified_columns = set(categorical_columns + ordinal_columns + continuous_columns + [target_column])
    if specified_columns != set(all_columns):
        missing_cols = list(set(all_columns) - specified_columns)
        raise ValueError(f"The following columns are not specified: {missing_cols}")

    # Convert column names to indices
    categorical_indices = [train.columns.get_loc(col) for col in categorical_columns]
    ordinal_indices     = [train.columns.get_loc(col) for col in ordinal_columns]
    continuous_indices  = [train.columns.get_loc(col) for col in continuous_columns]
    

    if not os.path.exists(f"{SAVE_PATH}/DP-dataset-test-val"):
        try:
            os.makedirs(f"{SAVE_PATH}/DP-dataset-test-val")
        except:
            pass

    save_path = f"{SAVE_PATH}/DP-dataset-test-val/"
    name = save_path+'split_test_val_dataset_seed_'+str(SEED)+'.csv'

    test.to_csv(name, index=True)


    if not os.path.exists(f"{SAVE_PATH}/DP-dataset-train/"):
        try:
            os.makedirs(f"{SAVE_PATH}/DP-dataset-train/")
        except:
            pass
    
    save_path = f"{SAVE_PATH}/DP-dataset-train/"
    name = save_path+'split_train_dataset_seed_'+str(SEED)+'.csv'

    train.to_csv(name, index=True)


    # Train a synthesizer
    nf = train.copy()
    print("\n##**Generating DP for seed " + str(SEED) + "**")

    for e in EPSILON:
        rounds = None
        epochs = None
        if DP_ALG.lower() == "dpctgan":
            epochs = 50
        if e > 1 and DP_ALG.lower() == "aim":
            rounds = 50

        print("\n###**Generating DP for seed " + str(SEED) + " epsilon " + str(e) + "**")
        synth = Synthesizer.create(DP_ALG, verbose=False, epsilon=e, **({"rounds": rounds} if rounds is not None else {"epochs": epochs} if epochs is not None else {})) 
        # if rounds is None else Synthesizer.create(DP_ALG, verbose=False, epsilon=e, rounds=rounds)
        synth.fit(
            nf, preprocessor_eps = e/2,
            categorical_columns  = categorical_columns,
            ordinal_columns      = ordinal_columns,
            continuous_columns   = continuous_columns,
        )


        print("###**Generating Samples for seed " + str(SEED) + " epsilon " + str(e) + "**")

        sample_data = None
        X_dp = None
        Y_dp = None
        try:
            sample_data = synth.sample(int(train.shape[0]))
            sample_data = pd.DataFrame(sample_data, columns=train.columns)
            X_dp = sample_data.drop(columns=[target_column], axis=1)
            Y_dp = sample_data[target_column]
        except:
            raise ValueError(f"Error on sampling data with epsilon {e}")

        print("###**Saving Samples for seed " + str(SEED) + " epsilon " + str(e) + "**")
        save_path = f"{SAVE_PATH}/DP-dataset-epsilon-" + str(e) + "/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        name = save_path+DP_ALG+'_synthetic_train_dataset_seed_'+str(SEED)+'_epsilon_'+str(e)+'.csv'

        dp_dataset = pd.concat([X_dp, Y_dp], axis=1)
        dp_dataset.to_csv(name, index=True)

