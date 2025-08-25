import os
import warnings
import pandas as pd

from typing import Callable, List, Any, Tuple
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from Benchmark.dataconf import BenchmarkDatasetConfig
from Benchmark.utils.types import IntOrTuple, DFTuple
from Benchmark.utils.verifiers import check_data_loader, check_splitdata, check_target, read_verification

DEFAULT_SEEDS = [5,42,253,4112,32645,602627,153073,53453,178753,243421,767707,113647,796969,553067,96797,133843,6977,460403,126613,583879]
DEFAULT_EPS   = [0.25, 0.5, 0.75, 1, 5, 10, 15, 20]
DP_ALGORITHM  = ""

class BenchmarkInfo:
    def __init__(self, dp_method:str, output_dir: str, data_loader: Callable[..., DFTuple] | None = None, 
                 split_data: IntOrTuple | None = None, normalize: bool = True, seeds: List[int] = [DEFAULT_SEEDS],
                 eps: List[float|int] = [DEFAULT_EPS], classifier: Any = None, classifier_kwargs: Any = None):
        """
        Set of possible confiigurations for the Benchmark experiments. 
        Parameters
        ----------
        dp_method : str
            Which DP symthetic data generator was used
        output_dir : str
            Directory to save the experiment logs and metrics.
        data_loader : Callable, optional
            In case a new data loader needs to be used, refer to the documentation to understand the default data loader's behaviour.
        split_data : IntOrTuple, optional
            Split distributions used while loading data. If not provided, the final distributions are **0.6, 0.2 and 0.2**, which is `split_data = (0.4, 0.5)`.
        normalize : bool, optional
            Allow MinMax normalization of the data. Default is **True**.
        seeds : List[int], optional
            List of seeds for the benchmark. Used to increase reproducibility.
        eps : List[float|int], optional
            List of DP epsilons (privacy budget) analysed during the benchmark.
        classifier : Any, optional
            Custom classifier. **Must implement fit, predict and predict_proba**. Default is [XGBoost](https://xgboost.readthedocs.io/en/stable/).
        classifier_kwargs : Any, optional
            Custom parameters for the classifier.
        
        """
        
        self.dp_method    = dp_method
        self.output_dir   = output_dir
        self.normalize    = normalize
        self.seeds = seeds
        self.eps   = eps

        global DP_ALGORITHM
        DP_ALGORITHM = self.dp_method 

        check_splitdata(split_data)
        self.split = split_data

        # Wrap user-supplied function with enforcement
        self.data_loader = check_data_loader(data_loader) if data_loader is not None else self.__data_loader

        self.classifier = classifier
        self.classifer_kwargs = classifier_kwargs

    def dataloader(self, **kwargs) -> DFTuple:
        """
        Data loader, by default assumes that within the `baseline_dir` there exists a CSV file with the name set in `filename` parameter.
        
        If the `split_data` has been set before, it will look for the file mentioned and split it into three sets following the provided distribution.
        
        The split happens sequentially, if two values has been provided to split, the first split (train+test) happens normally, and then the test set is split following the second distribution.
        
        If only one number has been provided and no test directory found, the split happens sequentially following the distribution of the test set.
      
        **Please refer to the documentation to understand how the default dataloader expects the directory structure to be like.**

        Parameters
        ----------
        data_conf : DatasetConf
            Configuration of the desired dataset.
        filename : str
            The name of the CSV file to load. 
        seed : int
            The current seed used to load the file and split the data.
        verbose : bool, optional
            If `true` prints information on the laoded dataset.
        extra_processing : Callable, optional
            Custom (users) porcessing function applied to loaded data. Will be called using kwargs and the loaded data as arguments.
        kwargs : Any, optional,
            If an extra processing function is provided, will be forwarded while calling, with the loaded dataset.
        
        data_path : str
            Path to the directory of the dataset files. (will be in data_conf)
            Ex: if your dataset named **D1** is located in `~/data/D1/D1.csv`, set it to `~/data/D1/`

        Returns
        ----------
        Three tuple[pd.DataFrame, pd.DataFrame]
            - A 2-tuple of pandas DataFrames `(X, y)`.
        """
        return self.data_loader(**kwargs)


    @check_data_loader
    def __data_loader(self, data_conf: BenchmarkDatasetConfig, filename: str, seed: int,  **kwargs) -> DFTuple:
        return __load_data(data_conf, filename, seed, self.split, **kwargs)



def __load_data(data_conf: BenchmarkDatasetConfig, filename: str, seed: int, verbose: bool=False, split: IntOrTuple | None = None, extra_processing: Callable | None = None, **kwargs) -> DFTuple:
    if verbose:
        print(f"** Loading dataset {data_conf.name.upper()} **")
    
    if split is None:
        split = (0.4, 0.5)

    test_path = f"{data_conf.dir}/{data_conf.name}/{DP_ALGORITHM}/dataset-test-val/"
    test_filename = f"{test_path}/{filename}_test_seed_{seed}.csv"

    ds = pd.read_csv(data_conf.dir + f"/{filename}", index_col=0, usecols=data_conf.usecols)
    
    # Verify if data was read successfully
    read_verification(ds, data_conf.usecols)

    # Apply extra processing to dataset if the user wants it
    if extra_processing is not None:
        extra_processing(ds, **kwargs)

    # Ensure all dataset is numerical
    for col in data_conf.categorical_cols:
        if not pd.api.types.is_numeric_dtype(ds[col]):
            ds[col] = ds[col].astype('category').cat.codes # Int encode

    X = ds.drop(columns=[data_conf.target])
    y = ds[data_conf.target]

    # Split data
    if not os.path.exists(test_path) or not os.path.exists(test_filename):
        if verbose:
            train_split_distrib = 1 - split[0] if isinstance(split, Tuple) else split
            val_split_distrib = split[0] * (1 - split[1]) if isinstance(split, Tuple) else split * (1 - split)
            test_split_distrib = split[0] * split[1] if isinstance(split, Tuple) else split * split
            print(f"[Info] Test directory and/or file with test set not found, the provided {filename} will be split into three sets with distributions {(train_split_distrib, val_split_distrib, test_split_distrib)}.")
        
        # No test path found, so split the data from filename
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split[0] if isinstance(split, Tuple) else split, random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=split[1] if isinstance(split, Tuple) else split, random_state=seed)

    else:
        X_train = X
        y_train = y

        test_ds = pd.read_csv(test_filename, index_col=0)
       
        # Verify if data was read successfully
        read_verification(test_ds, data_conf.usecols)

        # Apply extra processing to dataset if the user wants it
        if extra_processing is not None:
            extra_processing(test_ds, **kwargs)
            
        X_test = test_ds.drop(columns=[data_conf.target])
        y_test = test_ds[data_conf.target]

        if isinstance(split, Tuple):
            warnings.warn(f"You provided a tuple of splitting distribution and a test directory and file has been found in {test_path}, the second value of the tuple will be used.")

        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=split[1] if isinstance(split, Tuple) else split, random_state=seed)

    if verbose:
        data = [
            ["X_train", X_train.shape],
            ["X_val",   X_val.shape],
            ["X_test",  X_test.shape],
            ["y_train", y_train.shape],
            ["y_val",   y_val.shape],
            ["y_test",  y_test.shape],
        ]
        print("#### Data Split Information ####")
        print(tabulate(data, headers=["Dataset", "Shape"], tablefmt="github"))

    # Check that the target column is binary
    check_target(y_train, data_conf.target)
    check_target(y_val, data_conf.target)
    check_target(y_test, data_conf.target)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
