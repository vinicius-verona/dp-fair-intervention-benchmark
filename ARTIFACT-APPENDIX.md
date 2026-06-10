# Artifact Appendix

Paper title: Where to Intervene? Benchmarking Fairness-Aware Learning on Differentially Private Synthetic Tabular Data

Requested Badge(s):

- [ x ] **Available**
- [ x ] **Functional**
- [ x ] **Reproduced**

## Description

The artifact accompanying the paper **Where to Intervene? Benchmarking Fairness-Aware Learning on Differentially Private Synthetic Tabular Data**
consists of the complete benchmarking framework mentioned in the paper, used to evaluate
fairness-aware learning mechanisms on differentially private (DP)  synthetic tabular data.
The framework is organized into two independent modules:
`Data Generation` and `Benchmarking`, enabling reproducible experiments and straightforward extensibility.

> Data Generation Module:
The data generation modules provides an interface for producing DP synthetic data from arbitrary
tabular data source. Given an input dataset and a user-defined privacy budget, the module automatically
executes the selected DP synthesizer and generates one or more synthetic datasets preserving the original schema.
The current version supports both AIM and MST data synthesizers and accept other synhtesizers using common interface.

> Benchmarking Module:
This module provides a generic evaluation framework for investigating utility, privacy and fairness trade-offs.
It accepts any dataset, considering it has at least one untouched dataset, as well as a DP synthetic dataset,
originated as a result of the Data Generation module. It also provides a way to integrate any machine learning classifier
that implements the method `train()`, `predict()`, and `predict_proba()`.

The abstraction and independence of both modules facilitate the generation of DP synthetic datasets,
it also enables benchmarking across a wide range of machine learning algorithms. Furthermore, the modular
approach between data generation and benchmarking enables reproducible experiments while simplifying the
integration of other datasets, synthesizers, and classifiers.

### Security/Privacy Issues and Ethical Concerns

This artifact does not introduce additional security risks to the evaluator's machine.

This framework does not collect, send, transmit or share any user data outside the local environment.
The artifact depends on several open-source third-party Python libraries. As with any software dependency,
evaluators should follow standard software supply-chain security practices.

The benchmark relies exclusively on publicly available datasets and DP synthetic dataset used for research purposes.
No ethical review board approval was required given the public nature of the datasets.

Nonetheless, the framework evaluate fairness on potentially sensitive attributes, (i.e. gender and race)
Thus every analysis and results must be interpreted carefully and within social and legal context of the application.

## Basic Requirements

### Hardware Requirements

**Minimal hardware Requirements**: Although experiments can be executed on a laptop, we highly recommend machines with High CPU compute Power, High RAM (≥ 64Gb) and High Storage.

Experiments in this paper were run on the Grid500 cluster machines in Nancy and Grenoble sites.

The precise configurations can be found in https://www.grid5000.fr/w/Nancy:Hardware and https://www.grid5000.fr/w/Grenoble:Hardware

### Software Requirements

**OS used during experiments**: Debian GNU/Linux 11 (bullseye)

**Python**: Python 3.9 or less than 3.13 (tested on Python 3.9–3.12).

The TensorFlow dependency imposes a limitation on the Python version.

**Package manager**: `pip` (standard). No container runtime is required for this artifact.

The exact dependency list is specified in the `pyproject.toml` file at the root of the repository.

**Datasets**: All datasets used in the paper are either included in the repository under `data/` or downloaded automatically at runtime:

- **Adult** (UCI Census Income) — included under `data/`
- **COMPAS** — included under `data/`
- **ACSIncome** (Utah subset, 2018) — downloaded automatically via the `folktables` library
- **BiasOnDemand** — generated programmatically via the BoD generator at experiment time

No ML models need to be downloaded separately for the evaluation of the artifact; all classifiers are instantiated from scikit-learn, XGBoost, and standard libraries at runtime when installing the framework.

### Estimated Time and Storage Consumption

All our experiments have been executed on the Grid5000 cluster, where we are able to run multiple instances in parallel. Due to the remote execution and the parallel, modular approach (separating data generation from the benchmark itself), only a rough estimate can be provided. 

Based on our interactions with our cluster, we expect it will take, on average, 4 days per dataset to complete the full set of experiments. This time can be reduced or extended depending on the available computer power.    

## Environment

### Accessibility

The artifact is publicly accessible via GitHub and PyPI:

- **GitHub repository** (source code, datasets, notebooks, examples):
https://github.com/vinicius-verona/dp-fair-intervention-benchmark/tree/dev
- **PyPI package** (latest release, v0.2.0):
https://pypi.org/project/BenchmarkDPFair/

The artifact evaluators should use the GitHub repository as the primary reference for the full codebase, datasets, and example scripts. The PyPI package provides the installable library with no ablation study.

### Set up the environment

Our recommended installation method is via PyPI. With a Python 3.9+ environment:

```bash
pip install BenchmarkDPFair
```

Alternatively, there is the possibility to install from source, which also gives access to the `data/`, `example/`, and `notebook/` directories (be aware that these may consume more resources on your disk given the data provided):

```bash
git clone https://github.com/vinicius-verona/dp-fair-intervention-benchmark.git
cd dp-fair-intervention-benchmark
pip install -e .
```

For both, we recommend using a dedicated virtual environment to avoid dependency conflicts:

```bash
python3 -m venv dpfair-env
source dpfair-env/bin/activate   # On Windows: dpfair-env\\Scripts\\activate
# Installation approach via PyPi or source installation
```

The `private-pgm` dependency is fetched directly from GitHub if instsalled from source at install time (pinned commit).
When installing from PyPi, the latest version will be used.

**Expected result**: Installation completes without errors and the `BenchmarkDPFair` package is importable.

### Testing the Environment (Required for Functional and Reproduced badges)

After installation, a quick dummy test  can also be run using the provided example script, which runs a minimal experiment (two seeds, and two reduced privacy budget) on the COMPAS dataset:

```bash
mkdir -p dummy-test/data/Compas

cd dummy-test

curl -L https://raw.githubusercontent.com/vinicius-verona/dp-fair-intervention-benchmark/dev/data/Compas/compas.csv -o ./data/Compas/compas.csv

curl -L https://raw.githubusercontent.com/vinicius-verona/dp-fair-intervention-benchmark/dev/example/dummy.py -o ./dummy.py

python3 dummy.py --seeds 1 2
```

**Expected output**: The script prints progress messages for each seed/synthesizer combination during data generation, followed by benchmark results.
No exceptions should be raised. Output CSV files with fairness and utility metrics will be written to `./output/Dummy-Compas/`. The run should complete in a few minutes for the minimal configuration.
Some warning may raise concerning GPU or DataFrame, these are harmless.

To remove the GPU warning, you may export three environment variables

```
TF_CPP_MIN_LOG_LEVEL=3 TF_ENABLE_ONEDNN_OPTS=0 PYTHONWARNINGS=ignore
```

## Artifact Evaluation

### Main Results and Claims

### Main Result 1: Fairness interventions under DP+Fair partially recover fairness degradation

When fairness mechanisms are applied to DP synthetic data (DP+Fair), group disparities introduced by DP are partially reduced across the metrics EOD and SPD.
This is statistically validated with Wilcoxon signed-rank tests (Table 1 in the paper), showing reduction in |EOD| and |SPD| under DP+Fair vs. DP-only across the four main datasets and classifiers. MAD is not globally improved.

### Main Result 2: Post-processing achieves the strongest fairness–utility trade-offs

Among the three intervention stages (pre-processing, in-processing, post-processing), post-processing methods — particularly ROC and EqOdds — consistently achieve the best fairness–utility trade-offs across privacy budgets, datasets, and classifiers. This is confirmed by the Pareto-front analysis in Figures 2–4 and verified statistically by the paired Wilcoxon tests.

## **Experiments**
### **Experiment 1: Default Experiments — All Findings**

- ***Overview****

A set of shell scripts is provided to facilitate experiment execution. These scripts are available at:

https://github.com/vinicius-verona/dp-fair-intervention-benchmark/tree/dev/example

- ***Execution Steps****

1. ****Download the dataset.**** Obtain the original CSV file for the desired dataset from:

https://github.com/vinicius-verona/dp-fair-intervention-benchmark/tree/dev/data

Place the file in the directory `./data/$DATASET/`, where `$DATASET` is the name of the chosen dataset. A pre-generated example of the expected directory structure is available at the same link.

2. ****Run data generation and benchmarking.**** Execute the shell script twice — once for data generation (`-option 2`) and once for benchmarking (`-option 1`). The example below runs both steps for Batch 1 of the COMPAS dataset:

```

./script.sh --option 2 --dataset Compas \

--number 1 --output-suffix "COMPAS-1-experiment-artifact-functional-DataGen"

./script.sh --option 1 --dataset Compas \

--number 1 --output-suffix "COMPAS-1-experiment-artifact-functional-Benchmark"

```

The script arguments are defined as follows:

- `-option`: Execution mode. Use `1` for benchmarking and `2` for data generation.

- `-number`: Batch index. Accepts values `1` through `5`.

- `-output-suffix`: A string appended to the generated log file names for identification.

3. ****Merge and plot results.**** Once execution is complete, merge the output files for each configuration (fixed dataset, ML model, and synthesizer). Merging scripts and a plotting notebook are available at:

https://github.com/vinicius-verona/dp-fair-intervention-benchmark/tree/dev/notebook

4. ****(Optional) Run Python scripts directly.**** To view verbose output, the underlying Python scripts can be executed individually. Only the desired seeds need to be specified. Run any script with the `h` flag to display usage instructions:

```

python3 DPF_DataGeneration-Compas.py -h

```

- ***Expected Output****

Successful execution will produce:

- Synthetic train and test datasets for seeds `5`, `602627`, `767707`, and `133843`, and privacy budgets ε ∈ {0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 5, 10, 15, 20}, stored under `./data/Compas/aim/` and `./data/Compas/mst/`.
- Benchmark results for the same configuration (dataset, seed, and ε), stored under `./output/Compas/`.
- ***Runtime****

Assuming sufficient computational resources (CPU cores and RAM), the full experiment is expected to complete within ****3–4 days****.

- ***Supported Claims****

This experiment validates the main findings presented in ****Figures 2 and 3**** for the **COMPAS** dataset.

- ***Warning***
In case you have chosen to clone the repository, remember to replace the `example/data` directory by the `data` directory in the root of the project.

## Limitations

Numerical reproducibility of every data reported in the paper cannot be guaranteed due
to randomness in third-party libraries used by the framework, even when a fixed random seed is set.
As an example, we have identified that the DP synthesizers, as implemented in SmartNoise-Synth, do not always produce
identical outputs across runs despite seed control.

As a result, **exact numerical reproduction** reported in the paper is not expected,
we suggest, therfore, the evaluators not to treat small quantitative differences as a failure of the artifact.

In contrast, what we expect to be reproducible and recommend to be used as the validation criteria is the
**qualitative, stage-level ordering** of fairness interventions and the **geometric structure** of the Pareto-front plots.

This consistency is the basis of the central claim of the paper and demonstrates to be robust to seed-level variation.
Individual runs may shift points slightly, but the relative ordering of intervention stages in the fairness–utility space should
remains stable.

## Notes on Reusability

Beyond reproducing the results of this paper, the BenchmarkDPFair framework is designed as a general-purpose research tool
for anyone studying the intersection of differential privacy, synthetic data, and algorithmic fairness.
Its modular architecture was tought to provide a straightforward way to adapt to new settings without modifying the core library.

**Using a different dataset.** Any tabular dataset can be plugged in by defining a `DatasetGeneratorConfig` and `BenchmarkDatasetConfig`
with the appropriate column names, target variable, sensitive attributes, and preprocessing logic.

**Adding a new DP synthesizer.** The DataGenerator module accepts synthesizer that follows the interface expected by `SmartNoise-Synth`.
Researchers aiming to evaluate domain-specific DP generators (e.g., DP-GANs) can plug them in the generation pipeline without touching the benchmarking module.

**Adding a new classifier.** Any classifier implementing fit(), predict(), and predict_proba() can be passed directly to `BenchmarkInfo`.

**Exploring new privacy regimes.** The privacy budget list passed to `BenchmarkInfo` (parameter `eps`) is fully configurable. The user can test diverse ranges, such as high privacy budgets (ε < 0.05).

In summary, `BenchmarkDPFair` is intended to serve as an extensible framework for the research community at the intersection of privacy-preserving machine learning and algorithmic fairness.