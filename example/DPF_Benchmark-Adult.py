from BenchmarkDPFair.Benchmark import BenchmarkDatasetConfig, BenchmarkInfo, benchmark

for synth in ["aim", "mst"]:
    benchmark_config = BenchmarkInfo(
        dp_method=synth,
        output_dir="./data/Adult/output/",
        seeds = [5,42,253,4112,32645,602627,153073,53453,178753,243421,767707,113647,796969,553067,96797,133843,6977,460403,126613,583879],
        eps = [.25, .5, .75, 1, 5, 10, 15, 20]
    )

    benchmark_dataset = BenchmarkDatasetConfig(
        name = "Adult",
        target= "income",
        root_dir="../data",
        sensitive_attr = "sex",
        index_col="Unnamed: 0",
        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income', 'age', 'hours-per-week'],
        sensitive_cols = ['race', 'sex'],
    )

    benchmark(benchmark_info=benchmark_config, data_conf=benchmark_dataset)