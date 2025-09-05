import os
import warnings
import pandas as pd
import numpy as np
import inspect

from typing import Callable, List, Any, Tuple
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from .dataconf import BenchmarkDatasetConfig
from .utils.types import FloatOrTuple, DFTuple
from .utils.verifiers import check_data_loader, check_splitdata, check_target, read_verification, check_dict

from .utils.benchmark import Benchmark
from .utils.auxiliar import save_experiment

DEFAULT_SEEDS = [5,42,253,4112,32645,602627,153073,53453,178753,243421,767707,113647,796969,553067,96797,133843,6977,460403,126613,583879]
DEFAULT_EPS   = [0.25, 0.5, 0.75, 1, 5, 10, 15, 20]
DP_ALGORITHM  = ""

class BenchmarkInfo:
    def __init__(self, dp_method:str, output_dir: str, data_loader: Callable[..., DFTuple] | None = None, dlkwargs: dict | set = {},
                 split_data: FloatOrTuple | None = None, normalize: bool = True, seeds: List[int] = [DEFAULT_SEEDS],
                 eps: List[float|int] = [DEFAULT_EPS], classifier: Any = None, classifier_kwargs: dict | set | None = None):
        """
        Set of possible confiigurations for the Benchmark experiments. 

        **In case you do not use our own generator, read the documentation first to understand how the benchmark expects the data to be organized.**

        Parameters
        ----------
        dp_method : str
            Which DP symthetic data generator was used
        output_dir : str
            Directory to save the experiment logs and metrics.
        data_loader : Callable, optional
            In case a new data loader needs to be used, refer to the documentation to understand the default data loader's behaviour. data_loader must accept seed as an argument and also kwargs.
        dlkwargs : dict | set, optional
            Custom parameters for the data loader.
        split_data : FloatOrTuple, optional
            Split distributions used while loading data. If not provided, the final distributions are **0.6, 0.2 and 0.2**, which is `split_data = (0.4, 0.5)`.
        normalize : bool, optional
            Allow MinMax normalization of the data. Default is **True**.
        seeds : List[int], optional
            List of seeds for the benchmark. Used to increase reproducibility.
        eps : List[float|int], optional
            List of DP epsilons (privacy budget) analysed during the benchmark.
        classifier : Any, optional
            Custom classifier. **Must implement fit, predict and predict_proba**. Default is [XGBoost](https://xgboost.readthedocs.io/en/stable/).
        classifier_kwargs : dict | set, optional
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
        self.custom_loader = False if data_loader is None else True
        self.dlkwargs = dlkwargs

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
        
        Returns
        ----------
        Three tuple[pd.DataFrame, pd.DataFrame]
            - A 2-tuple of pandas DataFrames `(X, y)`.
        """
        return self.data_loader(**kwargs)


    @check_data_loader
    def __data_loader(self, data_conf: BenchmarkDatasetConfig, filename: str, seed: int,  **kwargs) -> DFTuple:
        return _load_data(data_conf, filename, seed, split=self.split, **kwargs)


def _load_data(data_conf: BenchmarkDatasetConfig, filename: str, seed: int, epsilon: float | None, verbose: bool=False, split: FloatOrTuple | None = None, extra_processing: Callable | None = None, **kwargs) -> DFTuple:
    
    if verbose:
        print(f"** Loading dataset {data_conf.name.upper()} **")
    
    if split is None:
        split = (0.4, 0.5)

    base, ext = os.path.splitext(filename)
    base_pattern = base.rsplit("_", 1)

    if (os.path.dirname(filename)):
        test_path = os.path.dirname(os.path.dirname(filename)) + "DP-dataset-test-val/"
    else:
        test_path = f"{data_conf.dir}/{data_conf.name}/{DP_ALGORITHM}/DP-dataset-test-val/"
        filename = f"{data_conf.dir}/{data_conf.name}/{DP_ALGORITHM}/DP-dataset-{f'epsilon-{epsilon}' if epsilon is not None else 'train'}/{filename}"

    test_filename = f"{base_pattern[0]}_test{ext}"

    cols = list(dict.fromkeys(data_conf.usecols + data_conf.index_col if data_conf.index_col else data_conf.usecols))
    ds = pd.read_csv(filename, usecols=cols)

    if data_conf.index_col:
        ds.set_index(data_conf.index_col, inplace=True)
    
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
    if not os.path.exists(test_path) or not os.path.exists(test_path + "/" + test_filename):
        if verbose:
            train_split_distrib = 1 - split[0] if isinstance(split, Tuple) else split
            val_split_distrib = split[0] * (1 - split[1]) if isinstance(split, Tuple) else split * (1 - split)
            test_split_distrib = split[0] * split[1] if isinstance(split, Tuple) else split * split
            print(f"[WARN] Test directory and/or file with test set not found, the provided {filename} will be split into three sets with distributions {(train_split_distrib, val_split_distrib, test_split_distrib)}.")
            print(f"       This is the path we are looking for: {test_path + "/" + test_filename}.\n")

        # No test path found, so split the data from filename
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split[0] if isinstance(split, Tuple) else split, random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=split[1] if isinstance(split, Tuple) else split, random_state=seed)

    else:
        X_train = X
        y_train = y

        # test_ds = pd.read_csv(test_filename, index_col=0)
        cols = list(dict.fromkeys(data_conf.usecols + data_conf.index_col if data_conf.index_col else data_conf.usecols))
        test_ds = pd.read_csv(test_path + "/" + test_filename, usecols=cols)

        if data_conf.index_col:
            test_ds.set_index(data_conf.index_col, inplace=True)
    
        # Verify if data was read successfully
        read_verification(test_ds, data_conf.usecols)

        # Apply extra processing to dataset if the user wants it
        if extra_processing is not None:
            extra_processing(test_ds, **kwargs)
            
        X_test = test_ds.drop(columns=[data_conf.target])
        y_test = test_ds[data_conf.target]

        if isinstance(split, Tuple):
            print(f"[WARN] You provided a tuple {split} of splitting distribution and a test directory and file has been found in {test_path}, the second value of the tuple will be used.\n")

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
        print("\n#### Data  Information ####")
        print(tabulate(data, headers=["Dataset", "Shape"], tablefmt="github"))
        print("###########################\n")

    # Check that the target column is binary
    check_target(y_train, data_conf.target)
    check_target(y_val, data_conf.target)
    check_target(y_test, data_conf.target)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


############# Experiments #############
def _experiment(seed, dataset_conf: BenchmarkDatasetConfig, benchmark_info: BenchmarkInfo, savefile):
    np.random.seed(seed)
    output_dir = f"{benchmark_info.output_dir}/{dataset_conf.name}/{benchmark_info.dp_method}/results/"

    print(f"\n*********************** Fair-only - seed = {seed} ***********************\n")
    extra_kwargs = {
        "data_conf": dataset_conf,
        "filename": dataset_conf.name + f"_split_dataset_seed_{seed}_train.csv",
        "custom_loader": benchmark_info.custom_loader,
        "epsilon": None,
        "seed": seed
    }
    original_experiment = Benchmark(
        name="baseline", data_loader=benchmark_info.data_loader, 
        normalize=benchmark_info.normalize, seed=seed, dlkwargs=benchmark_info.dlkwargs, ekwargs = extra_kwargs
    )
    original_experiment.run()

    save_experiment(original_experiment, seed, filename=savefile, path=output_dir,synth=benchmark_info.dp_method)

    del original_experiment

    for epsilon in benchmark_info.eps:
        print(f"\n*********************** DP & DP+Fair | ε={epsilon} ***********************\n")
        extra_kwargs = {
            "data_conf": dataset_conf,
            "filename": dataset_conf.name + f"_split_dataset_seed_{seed}_epsilon-{epsilon}.csv",
            "custom_loader": benchmark_info.custom_loader,
            "epsilon": epsilon,
            "seed": seed
        }
        dp_experiment = Benchmark(
            name="dp", data_loader=benchmark_info.data_loader, 
            normalize=benchmark_info.normalize, seed=seed, dlkwargs=benchmark_info.dlkwargs, ekwargs=extra_kwargs
        )
        dp_experiment.run()

        save_experiment(dp_experiment, seed, epsilon, filename=savefile, path=output_dir,synth=benchmark_info.dp_method)

        del dp_experiment.data_loader, dp_experiment


def benchmark(data_conf: BenchmarkDatasetConfig, benchmark_info: BenchmarkInfo):
    """
    Execute benchmark of Fairness interventions on models trained on original data and differentially private synthetic data.
    
    **The results obtained are output into a csv file in the defined output directory.**

    Parameters
    -----------
    data_conf: BenchmarkDatasetConfig
        Configurations on the dataset used
    
    benchmark_info: BenchmarkInfo
        Configurations about the experiments
    """
    
    print(f"Running DP Benchmark on dataset: '{data_conf.name}' with target: '{data_conf.target}' and sensitive attribute: '{data_conf.sensitive_attr}'")

    savefile = f"benchmark_results_seeds_{'_'.join(str(seed) for seed in benchmark_info.seeds)}_eps_{'_'.join(str(e) for e in benchmark_info.eps)}_synth_{benchmark_info.dp_method}.csv"

    for seed in benchmark_info.seeds:
        _experiment(seed, data_conf, benchmark_info, savefile)