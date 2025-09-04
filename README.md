# DP+Fair Benchmarking Framework

This repository provides a Python framework for **benchmarking mechanisms** on **Differential Privacy data synthesizing** and **Fairness**.  
It has been developed as part of a research project at **Inria**, by *Vinicius Gabriel Angelozzi V. de R.* and *Héber H. Arcolezzi*.  

The framework is part of an academic paper and can be cited (see [Citation](#citation)).  

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
from Benchmark import BenchmarkDatasetConfig, BenchmarkInfo, benchmark
from DataGenerator import DatasetGeneratorConfig, generate_data
from sklearn.ensemble import RandomForestClassifier

# Generate Data
dataset_conf = DatasetGeneratorConfig()


# Dataset configuration
dataset_conf = BenchmarkDatasetConfig(
    target_column="label",
    categorical_columns=["gender", "race"],
    ordinal_columns=["education"],
    # Optional: custom dataloader
    # dataloader=my_custom_loader
)

# Benchmark configuration
benchmark_conf = BenchmarkInfo(
    output_dir="./results",
    data_dir="./data",
    seed=42,
    classifier=RandomForestClassifier
)

# Run benchmark
benchmark(dataset_conf, benchmark_conf)
```

More detailed examples can be found in the [`examples/`](examples/) directory.

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{Angelozzi2025WhereToIntervene,
  title={Where to Intervene? Benchmarking Fairness-Aware Learning on Differentially Private Synthetic Tabular Data [Experiment, Analysis & Benchmark]},
  author={Angelozzi V. de R. Vinicius Gabriel and H. Arcolezi Héber},
  conference={To-be-Anounced},
  year={2026},
  institution={Inria, Grenoble INP}
}
```

---

## License

License: **MIT**

---

## Contributing

Contributions are welcome:

* Open an issue for bug reports or feature requests
* Submit a pull request to the `main` branch for code contributions

---

## Contact

For questions regarding this framework, please contact:

* **Vinicius Gabriel Angelozzi Verona de Resende** — \[contact.verona@tutanota.com]
* **Héber Hwang Arcolezi** — \[heber.hwang-arcolezi@inria.fr]
