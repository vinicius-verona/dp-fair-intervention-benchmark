# DP+Fair Benchmarking Framework

This repository provides a Python framework for **benchmarking fairness mechanisms** on **Differentially Private Synthetic Data**.  


---

## Features

- ⚡ Simple, reproducible setup for benchmarking algorithms  
- 🧩 Flexible API to plug in any classifier implementing `fit`, `predict`, and `predict_proba`  
- 📊 Pre-offered datasets included under `data/`  
- 🔬 Configurable experiment settings: dataset schema, dataset synthesizer, seeds, privacy-budget, input/outputs, classifier, data pre-processing.  

---

## Installation

To install, clone the repository and install dependencies:

```bash
git clone https://github.com/vinicius-verona/dp-fair-intervention-benchmark.git
cd dp-fair-intervention-benchmark
pip install -e .
```

Alternatively, you can install from **PyPI** (Yet to be made available):

```bash
pip install dp-fair-intervention-benchmark
````

---

## Repository Structure

```
├── data/         # Pre-offered datasets
├── src/          # Core source code
├── examples/     # Some demo
├── tests/        # Unit tests
└── README.md
```

---

## Quick Start

Here is a minimal usage example:

```python
from BenchmarkDPFair.DataGenerator import generate_data, DatasetGeneratorConfig
from BenchmarkDPFair.Benchmark import BenchmarkDatasetConfig, BenchmarkInfo

from sklearn.ensemble import RandomForestClassifier

# Generate Data
data_conf = DatasetGeneratorConfig(
    name = "Adult",
    target= "...",
    synthesizer = "aim",
    root_dir="./data",
    sensitive_attr = "...",
    categorical_cols = [...],
    sensitive_cols = [...],
    privacy_budgets=[...],
    binary_encoder=...
)

generate_data("adult.csv", data_conf, verbose=True) # Saves as CSV

# Dataset configuration
benchmark_config = BenchmarkInfo(
    dp_method="aim",
    output_dir="./data/Adult/output/",
    seeds = [...],
    eps = [...]
)

benchmark_dataset = BenchmarkDatasetConfig(
    name = "Adult",
    target= "income",
    root_dir="./data",
    sensitive_attr = "...",
    index_col="...",
    categorical_cols = [...],
    sensitive_cols = [...],
)

benchmark(benchmark_info=benchmark_config, data_conf=benchmark_dataset)
```

More detailed examples can be found in the [`example/`](example/) directory.

---

## License

License: **MIT**

---

## Contributing

Contributions are welcome:

* Open an issue for bug reports or feature requests
* Submit a pull request to the `main` branch for code contributions

---
