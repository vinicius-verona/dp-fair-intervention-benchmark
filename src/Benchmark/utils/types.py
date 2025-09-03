import pandas as pd
from typing import Callable, Tuple, List 


# Return type of data loader is either ((X_train, y_train), (X_test, y_test)) or ((X_train, y_train), (X_val, y_val), (X_test, y_test)) 
# Inner pair: always 2 DataFrames
DFPair = Tuple[pd.DataFrame, pd.DataFrame]

# Outer tuple: either 2 or 3 such pairs
TwoPairs   = Tuple[DFPair, DFPair]
ThreePairs = Tuple[DFPair, DFPair, DFPair]
DFTuple = TwoPairs | ThreePairs  # final type


FloatOrTuple = float | Tuple[float, float] 