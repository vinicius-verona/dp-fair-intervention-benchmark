# -*- coding: utf-8 -*-
"""
Generate-Adult-DP-Seeds-Epsilons
"""

import random
from typing import Callable, List, Optional
import numpy as np
import pandas as pd

from .utils.verifiers import read_verification
from sklearn.model_selection import train_test_split
from .dataconf import DatasetGeneratorConfig

from snsynth import Synthesizer
import os

def _default_binary_encoder(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Transform sensitive columns into binary columns by analysing frequency of each value.
    
    Parameters
    ----------

    df: pandas DataFrame    
        Dataframe to be transformed
    columns: List[str]
        Lists to apply binary transformation
    """
    for col in columns:
        most_common_value = df[col].mode()[0]
        df[col] = (df[col] == most_common_value).astype(int)
    return df

def _default_pre_process_dataset(X, y, binary_encoder : Optional[Callable[..., pd.DataFrame]] = None, sensitive_columns : List[str] = [], compressor : Optional[Callable[..., pd.DataFrame]] = None):
    """
    Default pre-processor, starts by encoding all non-numerical columns into numerical columns and then apply a binarization of the sensitive columns.
    The binarization may be changed by providing a different function.

    Paramters
    ---------
    X: pandas DataFrame 
        Dataframe with all features except the target column.
    y: pandas DataFrame
        Dataframe with the target column only.
    binary_encode: Callable[..., pd.DataFrame]
    sensitive_columns: List[str]
    compressor: Callable[..., pd.DataFrame]
        Function to apply a compression to the dataset, in order to reduce the cardinality of some categorical columns. This is useful for some synthesizers that do not handle well high-cardinality categorical columns. The function should receive a dataframe and return a dataframe with the same columns, but with some values merged together.
    """
    ds = pd.concat([X, y], axis=1)
    
    # Apply a compression to the dataset if a compression function is provided.
    if compressor is not None:
        ds = compressor(ds)
    
    for col in ds.columns:
        if not pd.api.types.is_numeric_dtype(ds[col]):
            ds[col] = ds[col].astype('category').cat.codes.astype("int64")

    # Binary encoding of sensitive columns, where 0 is the value in the category with the highest value
    if binary_encoder is None:
        binary_encoder = _default_binary_encoder


    ds = binary_encoder(ds, sensitive_columns)
    
    return ds

def generate_data(filename: str, test_filename: str,  data_conf: DatasetGeneratorConfig, path: Optional[str] = None, verbose:bool = False, **skwargs):
    """
    Generate differentially private synthetic data based on `filename` and as configured `data_conf`.

    Parameters
    ----------
    filename: str 
        The file containing the data to be synthesized
    data_conf: DatasetGeneratorConfig 
        The configuration of the generator
    path: str, optional
        Path to the file
    verbose : bool, optional
        Enable prints
    **skwargs
        Kwargs for the synthesizer object creation
    """
    random.seed(data_conf.seed)
    np.random.seed(data_conf.seed)

    # fetch dataset
    file_path = path + "/" if path is not None else data_conf.dir + "/" + data_conf.name + "/"

    if verbose:
        print(f"\n\n***************************")
        print(f"** Start Data Generation **")
        print(f"***************************")
        print(f"- Filename: {filename}")
        print(f"- Path: {file_path}")
        print(f"- Seed: {data_conf.seed}")
        print(f"- DP Data Synthesizer: {data_conf.synthesizer}")
        print(f"- Privacy budget: {data_conf.privacy_budgets}")
        print(f"- Sensitive Attribute: {data_conf.sensitive_attr}")
        print(f"- Use custom filter: {data_conf.filter is not None}")
        print(f"- Use custom pre-processer: {data_conf.pre_processing is not None}")
        print(f"- Train-Cal-Test distribution: {(1 - data_conf.test_split_size - data_conf.cal_split_size, data_conf.cal_split_size, data_conf.test_split_size)}")
        print(f"- Columns: ")
        print(f"     * Sensitive   = {data_conf.sensitive_cols}")
        print(f"     * Categorical = {data_conf.categorical_cols}")
        print(f"     * Ordinal     = {data_conf.ordinal_cols}")
        print(f"     * Continuous  = {data_conf.continuous_cols}")
        print(f"     * Index Col   = {data_conf.index_col}")
        print("\n")
        

    dataset = pd.read_csv(file_path + filename, usecols=data_conf.usecols)
    dataset_test = pd.read_csv(file_path + test_filename, usecols=data_conf.usecols) if test_filename != "" else pd.DataFrame()

    if data_conf.index_col is not None:
        dataset.set_index(data_conf.index_col, inplace=True)
        if test_filename != "":
            dataset_test.set_index(data_conf.index_col, inplace=True)


    if data_conf.filter is not None and isinstance(data_conf.filter, Callable):
        if verbose:
            print("[Info] Apply filtering to dataset")
        dataset = data_conf.filter(dataset)
        if test_filename != "":
            dataset_test = data_conf.filter(dataset_test)
    
    df_X = dataset.drop(columns=[data_conf.target], axis=1)
    df_y = dataset[[data_conf.target]]
    
    if test_filename != "":
        df_test_X = dataset_test.drop(columns=[data_conf.target], axis=1)
        df_test_y = dataset_test[[data_conf.target]]
    else:
        df_test_X = pd.DataFrame()
        df_test_y = pd.DataFrame()

    # Remove null values from dataset and its respective label
    null_indices = df_X[df_X.isnull().any(axis=1)].index

    # Drop those indices from both X and y
    df_X = df_X.drop(null_indices)
    df_y = df_y.drop(null_indices)

    if test_filename != "":
        # Remove null values from dataset and its respective label
        null_indices = df_test_X[df_test_X.isnull().any(axis=1)].index

        # Drop those indices from both X and y
        df_test_X = df_test_X.drop(null_indices)
        df_test_y = df_test_y.drop(null_indices)


    if verbose:
        print("[Info] Start pre-processing dataset")

    # Preprocess the dataset
    pre_processing = _default_pre_process_dataset
    preproc_kwargs = {
        "binary_encoder": data_conf.binary_encoder,
        "sensitive_columns": data_conf.sensitive_cols,
        "compressor": data_conf.compressor if data_conf.compressor is not None else None
    }

    if data_conf.pre_processing is not None:
        pre_processing = data_conf.pre_processing

    test_size = data_conf.test_split_size if data_conf.test_split_size is not None else 0.2

    if test_filename != "":
        df_X = pd.concat([df_X, df_test_X], axis=0)
        df_y = pd.concat([df_y, df_test_y], axis=0)
    
    ds          = pre_processing(df_X, df_y, **(preproc_kwargs if data_conf.pre_processing is None else {}))
    train, test = train_test_split(ds, test_size=test_size, random_state=data_conf.seed)

    # Ensure all columns are accounted for
    read_verification(ds, [x for x in data_conf.usecols if x != data_conf.index_col])
    synth_name = data_conf.synthesizer if isinstance(data_conf.synthesizer , str) else data_conf.synthesizer_name
    save_path = data_conf.dir + "/" + data_conf.name + "/" + synth_name + "/"

    if not os.path.exists(f"{save_path}/DP-dataset-test"):
        try:
            os.makedirs(f"{save_path}/DP-dataset-test")
        except:
            pass

    name = save_path+f'DP-dataset-test/{data_conf.name}_split_dataset_seed_'+str(data_conf.seed)+'_test.csv'
    test.to_csv(name, index=True)

    # Save the calibration and training datasets.
    if not os.path.exists(f"{save_path}/DP-dataset-train/"):
        try:
            os.makedirs(f"{save_path}/DP-dataset-train/")
        except:
            pass
    
    name = save_path+f'DP-dataset-train/{data_conf.name}_split_dataset_seed_'+str(data_conf.seed)+'_train.csv'
    train.to_csv(name, index=True)

    # Train a synthesizer
    nf = train.copy()

    for e in data_conf.privacy_budgets:
        if verbose:
            print(f"[Info] Start DP Data Synthesizer {data_conf.synthesizer.upper()} with budget {e}")

        synth = Synthesizer.create(data_conf.synthesizer_name, epsilon=e, **skwargs) if isinstance(data_conf.synthesizer, str) else data_conf.synthesizer(epsilon=e, **skwargs)

        synth.fit(
            nf, preprocessor_eps = e/2,
            categorical_columns  = data_conf.categorical_cols,
            ordinal_columns      = data_conf.ordinal_cols,
            continuous_columns   = data_conf.continuous_cols,
        )

        sample_data = None
        X_dp = None
        Y_dp = None
        try:
            sample_data = synth.sample(int(train.shape[0]))
            sample_data = pd.DataFrame(sample_data, columns=train.columns)
            X_dp = sample_data.drop(columns=[data_conf.target], axis=1)
            Y_dp = sample_data[data_conf.target]
        except:
            raise ValueError(f"Error on sampling data with epsilon {e}")

        if verbose:
            print(f"[Info] Saving DP Synthesized data")

        print("###**Saving Samples for seed " + str(data_conf.seed) + " epsilon " + str(e) + "**")
        dp_save_path = f"{save_path}/DP-dataset-epsilon-" + str(e) + "/"
        if not os.path.exists(dp_save_path):
            os.makedirs(dp_save_path)
        
        dp_dataset = pd.concat([X_dp, Y_dp], axis=1)
        dp_train_set = dp_dataset.copy()
        
        name = dp_save_path+data_conf.name+'_split_dataset_seed_'+str(data_conf.seed)+'_epsilon-'+str(e)+'.csv'
        dp_train_set.to_csv(name, index=True)
        
        del synth, dp_dataset, name, X_dp, Y_dp, sample_data, dp_train_set#, dp_cal_set

