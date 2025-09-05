from BenchmarkDPFair.Benchmark import BenchmarkDatasetConfig, BenchmarkInfo, benchmark


benchmark_config = BenchmarkInfo(
    dp_method="aim",
    output_dir="./data/Adult/output/",
    seeds = [42],
    eps = [.25]
    # dp_method:str, output_dir: str, data_loader: Callable[..., DFTuple] | None = None, dlkwargs: dict | set | None = None,
    # split_data: FloatOrTuple | None = None, normalize: bool = True, seeds: List[int] = [DEFAULT_SEEDS],
    # eps: List[float|int] = [DEFAULT_EPS], classifier: Any = None, classifier_kwargs: dict | set | None = None
)

benchmark_dataset = BenchmarkDatasetConfig(
    name = "Adult",
    target= "income",
    root_dir="./data",
    sensitive_attr = "sex",
    index_col="Unnamed: 0",
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income', 'age', 'hours-per-week'],
    sensitive_cols = ['race', 'sex'],
)

benchmark(benchmark_info=benchmark_config, data_conf=benchmark_dataset)