from typing import List

class BenchmarkDatasetConfig:
    def __init__(self, name : str, target : str, sensitive_attr : str, categorical_cols : List[str] = [],
                 ordinal_cols : List[str] = [], continuous_cols : List[str] = [], root_dir : str = "../../data/", usecols : List[str] | None = None):
        """
        Configuration of a given dataset.

        Parameters
        ----------
        name : str
            Name of the dataset, this will be used for outputing logs
        dir : str
            Path to the root directory of the dataset. For example "../../data/" for the Adult dataset already provided.
        baseline_data_dir : str
            Directory where we find the original (non-dp) data.
        dp_data_dir : str
            Directory where we find the dp synthetic data.
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
        self.baseline_dir = baseline_data_dir
        self.dp_dir = dp_data_dir
        self.target  = target
        self.sensitive_attr   = sensitive_attr
        self.categorical_cols = categorical_cols
        self.ordinal_cols     = ordinal_cols 
        self.continuous_cols  = continuous_cols
        
        if usecols is not None and len(usecols) == 0:
            usecols = None

        self.usecols = usecols

        if not name:
            raise ValueError(f"Argument 'name' must not be empty as it is necessary for the benchmark.")
        
        if not root_dir:
            self.dir = "./"
        
        if not sensitive_attr:
            raise ValueError(f"A sensitive attribute is required for the benchmark.")
        
        if len(categorical_cols) == 0 and len(ordinal_cols) == 0 and len(continuous_cols) == 0:
            raise ValueError(f"The columns must be of one of the three categories: Categorical, Ordinal or Continuous.")
        
    def __str__(self):
        return f"BenchmarkDatasetConfig(name={self.name},dir={self.dir},target={self.target},sensitive_attr={self.sensitive_attr},categorical_cols={self.categorical_cols},ordinal_cols={self.ordinal_cols},continuous_cols={self.continuous_cols})"