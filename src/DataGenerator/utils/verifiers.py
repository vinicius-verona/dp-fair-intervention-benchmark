from typing import Callable
from functools import wraps
import pandas as pd


def check_transformer(func: Callable) -> Callable:
    """Decorator used to enforce data_loader returns the correct tuples of DataFrames."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> pd.DataFrame:
        result = func(*args, **kwargs)

        # Must be tuple of length 2 or 3
        if not isinstance(result, pd.DataFrame):
            raise TypeError("Return must be a pandas DataFrame.")
        
        return result
    return wrapper


def read_verification(ds:pd.DataFrame, cols) -> bool:
    all_columns = ds.columns.tolist()

    # Ensure all columns are accounted for
    specified_columns = list(cols)

    if set(specified_columns) != set(all_columns):
        missing_cols = list(set(all_columns) - specified_columns)
        raise KeyError(f"The following columns are not present in the dataframe: {missing_cols}.")