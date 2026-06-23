# DP+Fair Benchmarking Framework

This repository provides a Python framework for **benchmarking fairness mechanisms** on **Differentially Private Synthetic Data**.  


---

## Features

- ⚡ Simple, reproducible setup for benchmarking algorithms  
- 🧩 Flexible API to plug in any classifier implementing `fit`, `predict`, and `predict_proba`  
- 📊 Pre-offered datasets included under [`data/`](https://github.com/vinicius-verona/dp-fair-intervention-benchmark/tree/dev/data)  
- 🔬 Configurable experiment settings: dataset schema, dataset synthesizer, seeds, privacy-budget, input/outputs, classifier, data pre-processing.  

---

## Installation

> From source
To install, either use PyPi to isntall it, or clone the repository and install dependencies:

```bash
git clone https://github.com/vinicius-verona/dp-fair-intervention-benchmark.git
cd dp-fair-intervention-benchmark

python3 -m venv dpfair-env
source dpfair-env/bin/activate 

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

> Using PyPi **(SUGGESTED)**
```bash
python3 -m venv dpfair-env
source dpfair-env/bin/activate 
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install BenchmarkDPFair
````

---

## Repository Structure

```
├── data/         # Pre-offered datasets
├── src/          # Core source code
├── example/      # Files used in the main paper and a dummy example.
└── README.md
```

---

## Quick Start

Here is a dummy example:

```python
import argparse
from typing import List, Union
from BenchmarkDPFair.DataGenerator import generate_data, DatasetGeneratorConfig
from BenchmarkDPFair.Benchmark import benchmark, BenchmarkInfo, BenchmarkDatasetConfig

from sklearn.linear_model import LogisticRegression

ESTIMATOR_PARAMS = {
    'max_iter': 10000,
    'solver': 'saga',
    'l1_ratio': 0.5,
    'C': 0.8
}

lr = LogisticRegression
classifiers = [lr]
ckwargs = [
    ESTIMATOR_PARAMS,
]
classifier_name = ["LR"]
combinations = [
    (0, 0),
    (0, 1),
]

synths = ["aim", "mst"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments of Data Generation for Adult")

    parser.add_argument(
        "--seeds", "-s",
        nargs="+",        # 1 or more values
        type=int          # convert automatically to int
    )

    args = parser.parse_args()
    seeds = args.seeds
    
    eps : List[Union[int,float]] = [.05, .1]

    for synthesizer in synths:
        for s in seeds:
            data_conf = DatasetGeneratorConfig(
                name = "Compas",
                target= "two_year_recid",
                synthesizer = synthesizer,
                root_dir="./data",
                sensitive_attr = "race",
                categorical_cols = ['race', 'score_text', 'c_charge_degree','age', 'sex', 'two_year_recid'],
                sensitive_cols = ['race', 'sex'],
                ordinal_cols = ['priors_count'],
                privacy_budgets=eps,
                binary_encoder=binary_encode,
                seed = s,
                test_split_size=0.4,
                data_filter = filter_compas
            )

            generate_data(f"compas.csv", "", data_conf, "./data", verbose=True)

    for clf_idx, syn_idx in combinations:
        classifier = classifiers[clf_idx]
        synth = synths[syn_idx]

        benchmark_config = BenchmarkInfo(
            dp_method=synth,
            output_dir=f"./output/Dummy-Compas/{classifier_name[clf_idx]}/",
            seeds=seeds,
            eps = eps,
            classifier=classifier,
            classifier_kwargs=ckwargs[clf_idx]
        )

        benchmark_dataset = BenchmarkDatasetConfig(
            name = "Compas",
            target= "two_year_recid",
            root_dir="./data",
            sensitive_attr = "race",
            index_col="Unnamed: 0",
            categorical_cols = ['race', 'score_text', 'c_charge_degree','age', 'sex', 'two_year_recid'],
            ordinal_cols=["priors_count"],
            sensitive_cols = ['race', 'sex'],
        )


        benchmark(benchmark_info=benchmark_config, data_conf=benchmark_dataset)
```

More detailed examples can be found in the [`example/`](https://github.com/vinicius-verona/dp-fair-intervention-benchmark/tree/dev/example) directory.

---

## License

License: **MIT**

---

## Contributing

Contributions are welcome:
* Open an issue for bug reports or feature requests

---
