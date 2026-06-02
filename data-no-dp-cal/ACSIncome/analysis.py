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


# from memory_profiler import profile
from sklearn.preprocessing import MinMaxScaler
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

EPSILON = [1]#[0.25, 0.5, 0.75, 1, 5, 10, 15, 20]
NORMALIZE = True
path = ''
DP_ALG = 'aim' #'aim'

# Sensitive columns
sensitive_columns = ['SEX', 'RAC1P']
sensitive_attr = 'RAC1P'
ordinal_columns = ['SCHL', 'AGEP']
continuous_columns = []

# Non-sensitive columns
non_sensitive_columns = ['AGEP','COW','SCHL','MAR','OCCP','POBP','RELP','WKHP']

# Target column
target_column = 'PINCP'
categorical_columns = ['COW','MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P','PINCP']
USED_FEATURES = ['AGEP','COW','MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P', 'PINCP']

def getMetrics(metric):
    DI         = metric.disparate_impact()
    ACC        = metric.accuracy()
    ACC_PRIV   = metric.accuracy(privileged=True)
    ACC_UNPRIV = metric.accuracy(privileged=False)
    PREC       = metric.precision()
    REC        = metric.recall()
    MAD        = metric.accuracy(privileged=False) - metric.accuracy(privileged=True)
    EOD        = metric.equal_opportunity_difference()
    TPR        = metric.true_positive_rate()
    FPR        = metric.false_positive_rate()
    TNR        = metric.true_negative_rate()
    FNR        = metric.false_negative_rate()
    SPD        = metric.statistical_parity_difference()
    EODD       = metric.equalized_odds_difference()

    del metric

    return {
        "DI": DI if (DI is not None and not np.isnan(DI)) else 'inf',
        "ACC": ACC if (ACC is not None and not np.isnan(ACC)) else 'inf',
        "ACC_PRIV": ACC_PRIV if (ACC_PRIV is not None and not np.isnan(ACC_PRIV)) else 'inf',
        "ACC_UNPRIV": ACC_UNPRIV if (ACC_UNPRIV is not None and not np.isnan(ACC_UNPRIV)) else 'inf',
        "PREC": PREC if (PREC is not None and not np.isnan(PREC)) else 'inf',
        "REC": REC if (REC is not None and not np.isnan(REC) ) else 'inf',
        "MAD": MAD if (MAD is not None and not np.isnan(MAD) ) else 'inf',
        "EOD": EOD if (EOD is not None and not np.isnan(EOD) ) else 'inf',
        "TPR": TPR if (TPR is not None and not np.isnan(TPR) ) else 'inf',
        "FPR": FPR if (FPR is not None and not np.isnan(FPR) ) else 'inf',
        "TNR": TNR if (TNR is not None and not np.isnan(TNR) ) else 'inf',
        "FNR": FNR if (FNR is not None and not np.isnan(FNR) ) else 'inf',
        "SPD": SPD if (SPD is not None and not np.isnan(SPD) ) else 'inf',
        "EODD": EODD if (EODD is not None and not np.isnan(EODD) ) else 'inf',
    }

def original_experiment(x_train, y_train, x_test, y_test, seed=42, normalize=True, threshold=.5):

    # if dataset is COMPAS, switch
    privileged_groups = [{sensitive_attr: 1}] # Ex: White
    unprivileged_groups = [{sensitive_attr: 0}] # Ex: Not white
    
    scaler = None
    if normalize:
        scaler = MinMaxScaler()

    if scaler is not None:
        cols = x_train.columns
        x_train = scaler.fit_transform(x_train)
        x_train = pd.DataFrame(x_train, columns=cols)
        x_test = scaler.transform(x_test)
        x_test = pd.DataFrame(x_test, columns=cols)
        
        
    model = XGBClassifier(objective='binary:logistic', random_state=seed)
    # model = LogisticRegression(**ESTIMATOR_PARAMS)
    model.fit(x_train, y_train)
    y_pred_prob = model.predict_proba(x_test)[:, 1]
    y_pred = (y_pred_prob >= threshold).astype(int)
        
    y_preds = pd.DataFrame(y_pred, columns=[target_column])
    
    # Reset the index
    y_preds = y_preds.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    og_dataset_test = pd.concat([x_test, y_preds], axis=1)
        
    
    og_dataset_test_pred = BinaryLabelDataset(df=og_dataset_test, label_names=[target_column], protected_attribute_names=[sensitive_attr], 
                                unprivileged_protected_attributes=unprivileged_groups)


    y_test = y_test.reset_index(drop=True)
    df_test = pd.concat([x_test, y_test], axis=1)
    df_test = BinaryLabelDataset(df=df_test, label_names=[target_column], protected_attribute_names=[sensitive_attr], 
                                unprivileged_protected_attributes=unprivileged_groups)


    og_classification_metrics = ClassificationMetric(df_test, og_dataset_test_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    og_metrics = getMetrics(og_classification_metrics)
    
    og_classification_metrics.dataset            = None
    og_classification_metrics.classified_dataset = None

    del scaler,  y_preds, model, x_train, y_train, x_test, y_test, og_dataset_test, y_pred, y_pred_prob, og_dataset_test_pred, df_test, og_classification_metrics

    return {
        "original_classification_metrics": og_metrics,
    }


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
if len(sys.argv) != 2:
    print("Syntax: analysis.py seed")
    os.abort()
    
SEED = int(sys.argv[1])

STATES = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
              'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
              'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
              'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
              'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']

print(f"Running dataset generator with seed {SEED}")
if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
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
        
    max_metric = {
        "MAD": {
            "val": 0,
            "state": ""
        },
        "SPD": {
            "val": 0,
            "state": ""
        },
        "EOD": {
            "val": 0,
            "state": ""
        }
    }
    for state in STATES:

        # Load full dataset
        data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
        data = data_source.get_data(states=[state])

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
        

        # Train a synthesizer
        nf = train.copy()
        print("\n##**Baseline for seed " + str(SEED) + f" and state {state}**")

        X_train = train.drop(columns=[target_column])
        y_train = train[target_column]
        X_test = test.drop(columns=[target_column])
        y_test = test[target_column]
        baseline = original_experiment(X_train, y_train, X_test, y_test, seed=SEED)
        
        MAD = baseline['original_classification_metrics']['MAD']
        SPD = baseline['original_classification_metrics']['SPD']
        EOD = baseline['original_classification_metrics']['EOD']

        if abs(MAD) > abs(max_metric["MAD"]["val"]):
            max_metric["MAD"]["val"] = MAD
            max_metric["MAD"]["state"] = state

        if abs(SPD) > abs(max_metric["SPD"]["val"]):
            max_metric["SPD"]["val"] = SPD
            max_metric["SPD"]["state"] = state

        if abs(EOD) > abs(max_metric["EOD"]["val"]):
            max_metric["EOD"]["val"] = EOD
            max_metric["EOD"]["state"] = state


        print(f"\tMAD: {baseline['original_classification_metrics']['MAD']}")
        print(f"\tSPD: {baseline['original_classification_metrics']['SPD']}")
        print(f"\tEOD: {baseline['original_classification_metrics']['EOD']}")
    

    print("Maximum values and states")

    print(f"MAD: {max_metric['MAD']['val']} - {max_metric['MAD']['state']}")
    print(f"SPD: {max_metric['SPD']['val']} - {max_metric['SPD']['state']}")
    print(f"EOD: {max_metric['EOD']['val']} - {max_metric['EOD']['state']}")

