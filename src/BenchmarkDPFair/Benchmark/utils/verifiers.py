import inspect
from typing import Callable, Tuple
from functools import wraps
import pandas as pd

from .types import DFTuple, FloatOrTuple

def check_data_loader(func: Callable) -> Callable:
    """Decorator used to enforce data_loader returns the correct tuples of DataFrames."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> DFTuple:
        result = func(*args, **kwargs)

        # Must be tuple of length 2 or 3
        if not isinstance(result, tuple) or len(result) not in (2, 3):
            raise TypeError("Return must be a tuple of 2 or 3 pairs of DataFrames.")

        for pair in result:
            if not (isinstance(pair, tuple) and len(pair) == 2):
                raise TypeError("Each element must be a tuple of exactly 2 DataFrames.")
            if not all(isinstance(df, pd.DataFrame) for df in pair):
                raise TypeError("Each element of pair must be a pandas DataFrame.")

        return result
    return wrapper

def check_signatures(func: Callable, kwargs_dict:dict|set):
    sig = inspect.signature(func)

    # if function accepts kwargs, send everything as parameters
    accept_kwarg = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    if accept_kwarg:
        return kwargs_dict
    else:
        # Match parameters
        valid_args = { k: v for k,v in kwargs_dict.items() if  k in sig.parameters }
        return valid_args

def check_splitdata(x: FloatOrTuple | None) -> FloatOrTuple | None:
    """Type checker for split_data argument"""
    if x is None:
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) for i in x):
        return x
    raise TypeError("split_data must be int, (int, int), or None")


def read_verification(ds:pd.DataFrame, cols) -> bool:
    all_columns = ds.columns.tolist()

    # Ensure all columns are accounted for
    specified_columns = list(cols)

    if set(specified_columns) != set(all_columns):
        missing_cols = list(set(all_columns) - specified_columns)
        raise KeyError(f"The following columns are not present in the dataframe: {missing_cols}.")
    
def check_target(df : pd.DataFrame, target: str) -> bool:
    if target not in df.columns:
        raise ValueError(f"Column '{target}' not found in DataFrame.")
    
    unique_vals = df[target].dropna().unique()  # ignore NaNs
    return len(unique_vals) == 2

def check_dict(args : dict | set | None = None, key : str | None = None, ktype = None) -> Tuple[bool, str|None]:
    """Check if a dict has a key of certain type or just a key or if any key of a given type."""
    ret = False
    rkey = None

    if args is None:
        return False, None

    if key is not None:
        ret = key in args
        rkey = key

    if key is not None and ktype is not None:
        ret = key in args and isinstance(value, ktype)
        rkey = key

    if key is None and ktype is not None:
        for key, value in args.items():
            if isinstance(value, ktype):
                return True, key
        
    return ret, rkey