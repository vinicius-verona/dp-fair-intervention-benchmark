from typing import Callable, List
import pandas as pd

from .utils.verifiers import check_transformer

class DatasetGeneratorConfig:
    def __init__(self, name : str, target : str, synthesizer: str, sensitive_attr : str,  test_split_size: float = 0.4, categorical_cols : List[str] = [],
                 ordinal_cols : List[str] = [], continuous_cols : List[str] = [],
                sensitive_cols : List[str] = [], root_dir : str = "../../data/", usecols : List[str] | None = None,
                data_filter : Callable[..., pd.DataFrame] | None = None, binary_encoder : Callable[..., pd.DataFrame] | None = None, 
                pre_processer : Callable[..., pd.DataFrame] | None = None, privacy_budgets: List[int | float] = []):
        """
        Configuration of a given dataset.

        Parameters
        ----------
        name : str
            Name of the dataset, this will be used for outputing logs
        dir : str
            Path to the root directory of the dataset. For example "../../data/" for the Adult dataset already provided.
        target : str
            Column to be predicted and/or used as ground truth.
        sensitive_attr : str
            Senstive attribute in the dataset. So far, only one is possible. Ex: **race**.
        categorical_cols : List[str]
            Columns with categorical data.
        ordinal_cols : List[str]
            Columns with ordinal data.
        continuous_cols : List[str]
            Columns with continuous data.
        usecols : List[str], optional
            Columns to be read from the dataset file. If empty or none, all columns will be read.
        """

        self.name    = name
        self.dir     = root_dir
        self.target  = target
        self.sensitive_attr   = sensitive_attr
        self.sensitive_cols   = sensitive_cols
        self.categorical_cols = categorical_cols
        self.ordinal_cols     = ordinal_cols 
        self.continuous_cols  = continuous_cols
        self.synthesizer = synthesizer

        self.filter = check_transformer(data_filter) if data_filter is not None else None
        self.binary_encoder = check_transformer(binary_encoder) if binary_encoder is not None else None
        self.pre_processing = check_transformer(pre_processer) if pre_processer is not None else None
        self.privacy_budgets = privacy_budgets or [.25,.5,.75,1,5,10,15,20]

        if usecols is not None and len(usecols) == 0:
            usecols = None

        self.usecols = usecols
        self.split_size = test_split_size

        if not name:
            raise ValueError(f"Argument 'name' must not be empty as it is necessary for the benchmark.")
        
        if not root_dir:
            self.dir = "./"
        
        if not sensitive_attr:
            raise ValueError(f"A sensitive attribute is required for the benchmark.")
        
        if len(categorical_cols) == 0 and len(ordinal_cols) == 0 and len(continuous_cols) == 0:
            raise ValueError(f"The columns must be of one of the three categories: Categorical, Ordinal or Continuous.")
        
    def __str__(self):
        return f"DatasetGeneratorConfig(name={self.name},dir={self.dir},target={self.target},privacy_budgets={self.privacy_budgets},sensitive_attr={self.sensitive_attr},sensitive_cols={self.sensitive_cols},categorical_cols={self.categorical_cols},ordinal_cols={self.ordinal_cols},continuous_cols={self.continuous_cols})"