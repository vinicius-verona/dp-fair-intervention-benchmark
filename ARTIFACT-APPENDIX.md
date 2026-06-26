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
The framework is organised into two independent modules:
`Data Generation` and `Benchmarking`, enabling reproducible experiments and straightforward extensibility

> Data Generation Module:
The data generation module provides an interface for producing DP synthetic data from arbitrary
tabular data source. Given an input dataset and a user-defined privacy budget, the module automatically
executes the selected DP synthesiser and generates one or more synthetic datasets preserving the original schema.
The current version supports both AIM and MST data synthesisers and accepts other synthesisers using a common interface.

> Benchmarking Module:
This module provides a generic evaluation framework for investigating utility, privacy and fairness trade-offs.
It accepts any dataset, considering it has at least one untouched dataset, as well as a DP synthetic dataset,
originated as a result of the Data Generation module. It also provides a way to integrate any machine learning classifier
that implements the method `train()`, `predict()`, and `predict_proba()`.

The abstraction and independence of both modules facilitate the generation of DP synthetic datasets;
It also enables benchmarking across a wide range of machine learning algorithms. Furthermore, the modular
approach between data generation and benchmarking enables reproducible experiments while simplifying the
integration of other datasets, synthesisers, and classifiers.

### Security/Privacy Issues and Ethical Concerns

This artifact does not introduce additional security risks to the evaluator’s machine.

This framework does not collect, send, transmit or share any user data outside the local environment.
The artifact depends on several open-source third-party Python libraries. As with any software dependency,
evaluators should follow standard software supply-chain security practices.

The benchmark relies exclusively on publicly available datasets and DP synthetic dataset used for research purposes.
No ethical review board approval was required given the public nature of the datasets.

Nonetheless, the framework evaluates fairness on potentially sensitive attributes (i.e. gender and race)
Thus, every analysis and results must be interpreted carefully and within the social and legal context of the application.

## Basic Requirements

### Hardware Requirements

**Minimal hardware Requirements**:  Although experiments can be executed on a laptop, we highly recommend machines with High CPU compute Power, High RAM (≥ 64Gb), and High Storage.

Experiments in this paper were run on the Grid500 cluster machines in Nancy and Grenoble sites.

The precise configurations can be found in [https://www.grid5000.fr/w/Nancy:Hardware](https://www.grid5000.fr/w/Nancy:Hardware) and [https://www.grid5000.fr/w/Grenoble:Hardware](https://www.grid5000.fr/w/Grenoble:Hardware).

### Software Requirements

**OS used during experiments**: Debian GNU/Linux 11 (bullseye)

**Python**: Python 3.9 or less than 3.13 (tested on Python 3.9–3.12).

The TensorFlow dependency imposes a limitation on the Python version.

**Package manager**: `pip` (standard). No container runtime is required for this artifact.

The exact dependency list is specified in the `pyproject.toml` file at the repository root.

**Datasets**: All datasets used in the paper are included in the repository under `data/`:

- **Adult** (UCI Census Income) — included under `data/`
- **COMPAS** — included under `data/`
- **ACSIncome** (Utah subset, 2018) — included under `data/`
- **BiasOnDemand** — included under `data/`

No ML models need to be downloaded separately for the evaluation of the artifact; all classifiers are instantiated from scikit-learn, XGBoost, and standard libraries at runtime when installing the framework.

### Estimated Time and Storage Consumption

All our experiments have been executed on the Grid'5000 cluster, where we can run multiple instances in parallel. Due to the remote execution and the parallel, modular approach (separating data generation from the benchmark itself), only a rough estimate can be provided.

Based on our interactions with our cluster, we expect it will take, on average, 4 days per dataset to complete the full set of experiments. This time can be reduced or extended depending on the available computer power.

## Environment

### Accessibility

The artifact is publicly accessible via GitHub and PyPI:

- **GitHub repository** (source code, datasets, notebooks, examples):
   - https://github.com/vinicius-verona/dp-fair-intervention-benchmark/tree/dev
- **PyPI package** (latest release, v0.2.6):
   - https://pypi.org/project/BenchmarkDPFair/

The artifact evaluators should use the GitHub repository as the primary reference for accessing/dowloading one of the following: all code versions, datasets, configuration files, and example scripts required to reproduce the experiments presented in the paper. The repository contains both the implementation used for the main results and the additional code needed to reproduce the ablation studies, in their respective branches.

The PyPI package, in contrast, provides a simplified installation of the framework corresponding to the version used for the paper’s main results. While it offers a convenient way to reproduce the primary experiments, it does not include the modifications and supplementary components required to reproduce the ablation studies. Consequently, evaluators interested only in validating the main results are recommended to use the PyPI package for its simplicity, whereas those seeking to reproduce the complete set of experiments, including the ablations, should use the GitHub repository.

### Set up the environment

First, we strongly recommend creating a new Python environment before installing the package. This helps maintain a clean and reproducible setup, facilitates dependency and version management and update, and minimises potential conflicts with previously installed libraries.

```bash
python3 -m venv dpfair-env
source dpfair-env/bin/activate   # On Windows: dpfair-env\\Scripts\\activate
# Installation approach via PyPi or source installation
```

If you are a Windows or Linux user, please install PyTorch CPU-only dependencies: [Step not required for Mac users]
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Users on MacOS are required to install `llvm-openmp`, either using conda or brew, as below:

**Conda**:
```bash
conda install -c conda-forge llvm-openmp
```
**Brew**:
```bash
brew install libomp
```

Then, the library installation can proceed as per usual.

Our recommended method is via PyPI. With a Python 3.9+ and <3.13 environment:

```bash
pip install BenchmarkDPFair
```

Alternatively, there is the possibility to install from source, which also gives access to the `data/`, `example/`, and `notebook/` directories (be aware that these may consume more resources on your disk given the data provided):

```bash
git clone https://github.com/vinicius-verona/dp-fair-intervention-benchmark.git
cd dp-fair-intervention-benchmark
pip install -e .
```

**Expected result**: Installation completes without errors and the `BenchmarkDPFair` package is importable.

### Testing the Environment

After installation, a quick dummy test  can also be run using the provided example script, which runs a minimal experiment (two seeds and two reduced privacy budgets) on the COMPAS dataset:

```bash
mkdir -p dummy-test/data/Compas

cd dummy-test

curl -L https://raw.githubusercontent.com/vinicius-verona/dp-fair-intervention-benchmark/dev/data/Compas/compas.csv -o ./data/Compas/compas.csv

curl -L https://raw.githubusercontent.com/vinicius-verona/dp-fair-intervention-benchmark/dev/example/dummy.py -o ./dummy.py

python3 dummy.py --seeds 1 2 # You may choose any seed and amount of seeds you like, here, 1 and 2 are examples
```

**Expected output**: The script prints progress messages for each seed/synthesizer combination during data generation, followed by benchmark results.
No exceptions should be raised. Output CSV files with fairness and utility metrics will be written to `./output/Dummy-Compas/`. The execution should complete in a few minutes for the minimal configuration.
For a detailed verification that the execution was successful, following the example above, the following files should be found:
```text
data/
├── Compas/ # Dataset
│   ├── aim/
|   │   ├── DP-dataset-epsilon-0.1/
|   │   │   ├── Compas_split_dataset_seed_1_epsilon-0.1.csv # 3691 rows and 8 columns all integer values
|   │   │   └── Compas_split_dataset_seed_2_epsilon-0.1.csv # 3691 rows and 8 columns all integer values
|   │   ├── DP-dataset-epsilon-0.05/
|   │   │   ├── Compas_split_dataset_seed_1_epsilon-0.05.csv # 3691 rows and 8 columns all integer values
|   │   │   └── Compas_split_dataset_seed_2_epsilon-0.05.csv # 3691 rows and 8 columns all integer values
|   │   ├── DP-dataset-test/
|   │   │   ├── Compas_split_dataset_seed_1_test.csv # 2461 rows and 8 columns all integer values
|   │   │   └── Compas_split_dataset_seed_2_test.csv # 2461 rows and 8 columns all integer values
|   │   └── DP-dataset-train/
|   │       ├── Compas_split_dataset_seed_1_train.csv # 3691 rows and 8 columns all integer values
|   │       └── Compas_split_dataset_seed_2_train.csv # 3691 rows and 8 columns all integer values
│   ├── mst/
|   │   ├── DP-dataset-epsilon-0.1/
|   │   │   ├── Compas_split_dataset_seed_1_epsilon-0.1.csv # 3691 rows and 8 columns all integer values
|   │   │   └── Compas_split_dataset_seed_2_epsilon-0.1.csv # 3691 rows and 8 columns all integer values
|   │   ├── DP-dataset-epsilon-0.05/
|   │   │   ├── Compas_split_dataset_seed_1_epsilon-0.05.csv # 3691 rows and 8 columns all integer values
|   │   │   └── Compas_split_dataset_seed_2_epsilon-0.05.csv # 3691 rows and 8 columns all integer values
|   │   ├── DP-dataset-test/
|   │   │   ├── Compas_split_dataset_seed_1_test.csv # 2461 rows and 8 columns all integer values
|   │   │   └── Compas_split_dataset_seed_2_test.csv # 2461 rows and 8 columns all integer values
|   │   └── DP-dataset-train/
|   │       ├── Compas_split_dataset_seed_1_train.csv # 3691 rows and 8 columns all integer values
|   │       └── Compas_split_dataset_seed_2_train.csv # 3691 rows and 8 columns all integer values
│   └── compas.csv/
│
├── output/
│   └── Dummy-Compas/
│       └── LR/ # Classifier
│           └── Compas/ # Dataset
|               ├── aim/
|               │   └── results/
|               │       ├── log/
|               │       └── benchmark_results_seeds_1_2_eps_0.05_0.1_synth_aim.csv # 103 rows and 18 columns
|               │
|               ├── mst/
|                   └── results/
|                       ├── log/
|                       └── benchmark_results_seeds_1_2_eps_0.05_0.1_synth_mst.csv # 103 rows and 18 columns
│
└── dummy.py
```

Some warning messages related to GPU availability or DataFrame operations may be displayed during execution. These warnings are expected and can be safely ignored, as they do not affect the correctness of the results.

Additionally, warnings related to division-by-zero operations may appear for certain methods. These warnings are also expected and were taken into account during our analysis, as they correspond to specific cases that are subsequently identified and filtered during result processing.

## Artifact Evaluation

### Main Results and Claims

### Main Result 1: Fairness interventions under DP+Fair partially recover fairness degradation

When fairness mechanisms are applied to DP synthetic data (DP+Fair), group disparities introduced by DP are partially reduced across the EOD and SPD metrics.
This is statistically validated using Wilcoxon signed-rank tests (Table 1 in the paper), showing reductions in |EOD| and |SPD| under DP+Fair vs DP-only across the four main datasets and classifiers. MAD is not globally improved.

### Main Result 2: Post-processing achieves the strongest fairness–utility trade-offs

Among the three intervention stages (pre-processing, in-processing, post-processing), post-processing methods — particularly ROC and EqOdds — consistently achieve the best fairness–utility trade-offs across privacy budgets, datasets, and classifiers. This is confirmed by the Pareto-front analysis in Figures 2–4 and verified statistically by the paired Wilcoxon tests.

## **Experiments**
### **Experiment 1: Default Experiments — All Findings**

- ***Overview****

A set of Python and shell scripts is provided to facilitate experiment execution. These scripts are available at:

https://github.com/vinicius-verona/dp-fair-intervention-benchmark/tree/dev/example

- ***Execution Steps****

**Note**: The example hereby provided is intended to be executed on a computing cluster or a machine equipped with considerable computational resources.
Each script execution launches 4 seeds in parallel, which may exceed the resource constraints of a standard personal computer.
If sufficient resources are unavailable, sequential execution is recommended by either modifying the script or invoking the Python scripts manually.

1. ****Download/Copy the data directory.**** 

Obtain the original used datasets from:

https://github.com/vinicius-verona/dp-fair-intervention-benchmark/tree/dev/data

Place the directory within the same location as the execution scripts, i.e., `./example`, if you cloned the repository.
Verify that every `./data/$DATASET/` contains its own `$DATASET.csv` file, where `$DATASET` is the name of the chosen dataset, e.g. Compas.

The downloaded directory `./data` is a pre-generated example for the framework execution. All dataset files generated will be replaced once the framework is executed.

- ***Warning***

In case you have chosen to clone the repository, remember to replace the `example/data` directory with the `data` directory in the root of the project.

2. ****Run data generation and benchmarking.**** 

Execute the shell script twice — once for data generation (`-option 2`) and once for benchmarking (`-option 1`). 
The example below runs both steps for Batch 1 of the COMPAS dataset:

```bash
# Batch 1 = seeds 5 602627 767707 133843
batch=1; \
dataset=Compas; \
./script.sh \
  --option 2 \
  --dataset "$dataset" \
  --number "$batch" \
  --output-suffix "$dataset-$batch-experiment-artifact-functional-DataGen" && \
./script.sh \
  --option 1 \
  --dataset "$dataset" \
  --number "$batch" \
  --output-suffix "$dataset-$batch-experiment-artifact-functional-Benchmark"
```

If you want to execute the full batch of experiments, please refer to the note at the end of the section.

The script arguments are defined as follows:

* `-option`: Execution mode. Use `1` for benchmarking, `2` for data generation, and `3` for tracking execution via the process status command.
* `-number`: Batch index. Accepts values `1` through `5`; each executes 4 seeds in parallel.
* `-output-suffix`: A string appended to the generated log file names for identification.
* `--bod-combo`: This parameter should only be used when the chosen dataset is the BiasOnDemand. It accepts values from `1` to `6`, and it selects a specific BoD dataset combo. To replicate the main one in the paper, select the value `5`.

3. ****Plot results.**** 

**Note**: The texlive-full package must be installed

To generate the figures and tables associated with the Wilcoxon and Pareto analyses, execute the `plot.sh` script located in the `notebook` directory. This can be accomplished using the following commands:

```bash
sudo apt update && \
sudo apt install -y texlive-full && \
cd notebook && chmod u+x plot.sh && \
chmod u+x get-results.sh && ./plot.sh output  # Generates figures and tables for the main paper findings
```

For the provided pre-executed experiments, use:

```bash
sudo apt update && \
sudo apt install -y texlive-full && \
cd notebook && chmod u+x plot.sh && \
chmod u+x get-results.sh && ./plot.sh output-example  # Generates figures and tables for the main paper findings
```


For the ablation studies, use:

```bash
sudo apt update && \
sudo apt install -y texlive-full && \
cd notebook && chmod u+x plot.sh && \
chmod u+x get-results.sh && ./plot.sh ablation dp-split # Generates figures for the dp-split ablation study
```

A successful execution should generate at least 7 PDF files for each of the main datasets (Compas Adult and ACSIncome), 2 TeX files, 8 CSV files for claim 1, and 5 CSV files for claim 2. If some experiment has not been found, a message will be displayed on the form:
```
[SKIP] Results directory not found for ML=RF, Dataset=BoD-Config-5, Synth=aim, searched_dir=None
```

Below is an example of what the user should see upon running the script for the ablation study `dp-split`:
```
texlive-full is already installed.
Copying results to the current directory...
Copied: Adult
Copied: ACSIncome or downloaded automatically at runtime
Copied: Compas
Not found: /...../dp-fair-intervention-benchmark/notebook/../example/ablation/dp-split//BoD
Done.
Generating plots...
[RESULTS_ROOT] /...../dp-fair-intervention-benchmark/notebook
[DP_SYNTHESIZER] aim

================================================================================
Reading Claim 1 data for ML model: XGB
================================================================================
[READ] ML=XGB, Dataset=Adult, Synth=aim: 20 CSV files, 4420 rows
```

4. ****(Optional) Run Python scripts directly.****

 To view verbose output, the underlying Python scripts can be executed individually. Only the desired seeds need to be specified. Run any script with the `h` flag to display usage instructions:

```bash
python3 DPF_DataGeneration-Compas.py -h # Display the help menu for the Data Generation module
python3 DPF_Benchmark-Compas.py -h # Display the help menu for the Benchmark module
```

- ***Track Execution****

The script includes an option `3` to monitor which processes are currently executing for a selected dataset, displaying their associated commands.
Below are examples of execution and expected output:

```bash

./script --option 3 --dataset Compas
# Output Example

1198338  111  0.9 /..../dp-fair-intervention-benchmark/dpfair-env/bin/python3 DPF_DataGeneration-Compas.py -s 5
1198339  109  0.9 /..../dp-fair-intervention-benchmark/dpfair-env/bin/python3 DPF_DataGeneration-Compas.py -s 602627
1198340  107  0.9 /..../dp-fair-intervention-benchmark/dpfair-env/bin/python3 DPF_DataGeneration-Compas.py -s 767707
1198342  106  0.9 /..../dp-fair-intervention-benchmark/dpfair-env/bin/python3 DPF_DataGeneration-Compas.py -s 133843
1198561  0.0  0.0 /bin/bash ./script.sh --option 3 --dataset Compas
COMPAS processes under execution

```

- ***Expected Output****

Successful execution generates a directory structure similar to that reported in the dummy execution. So it produces:
1. Synthetic train and test datasets for seeds `5`, `602627`, `767707`, and `133843`, and privacy budgets ε ∈ {0.05, 0.1, 0.25, 0.5, 0.75, 1, 2, 3, 5, 10, 15, 20}, stored under `./data/Compas/aim/` and `./data/Compas/mst/`.
2. Benchmark results for the same configuration (dataset, seed, and ε), stored under `./output/Compas/`.

- ***Runtime****

Assuming sufficient computational resources (CPU cores and RAM), the full experiment is expected to complete within ****3–4 days****.

- ***Supported Claims****

This experiment validates the main findings presented in ****Figures 2 and 3**** for the **COMPAS** dataset.

- ***Full Experiments Replication****

The example provided above executes only a subset of the experiments presented in the paper. 
To replicate the complete set of experiments for a single dataset, re-execute the script 5 times, varying the `--number` parameter at each run (or the `$batch` variable in the given command example).

- ***All seeds used in the paper****

The following seeds were used across all datasets in the paper: 
```
5 602627 767707 133843 42 153073 113647 6977 253 53453 796969 460403 4112 178753 553067 126613 32645 243421 96797 583879
```

## Limitations

Numerical reproducibility of all the data reported in the paper cannot be guaranteed due
to randomness in third-party libraries used by the framework, even when a fixed random seed is set.
As an example, we have identified that the DP synthesisers, as implemented in SmartNoise-Synth, do not always produce
identical outputs across runs despite seed control.

As a result, the **exact numerical reproduction** reported in the paper is not expected,
We suggest, therefore, that the evaluators not treat small quantitative differences as a failure of the artifact.

In contrast, what we expect to be reproducible and recommend to be used as the validation criteria is the
**qualitative, stage-level ordering** of fairness interventions and the **geometric structure** of the Pareto-front plots.

This consistency is the basis of the paper's central claim and is robust to seed-level variation.
Individual runs may shift points slightly, but the relative ordering of intervention stages in the fairness–utility space should
remains stable.

## Notes on Reusability

Beyond reproducing the results of this paper, the BenchmarkDPFair framework is designed as a general-purpose research tool
for anyone studying the intersection of differential privacy, synthetic data, and algorithmic fairness.
Its modular architecture was thought to provide a straightforward way to adapt to new settings without modifying the core library.

**Using a different dataset.** Any tabular dataset can be plugged in by defining a `DatasetGeneratorConfig` and `BenchmarkDatasetConfig`
with the appropriate column names, target variable, sensitive attributes, and preprocessing logic.

**Adding a new DP synthesizer.** The DataGenerator module accepts synthesizer that follows the interface expected by `SmartNoise-Synth`.
Researchers evaluating domain-specific DP generators (e.g., DP-GANs) can plug them into the generation pipeline without altering the benchmarking module.

**Adding a new classifier.** Any classifier implementing fit(), predict(), and predict_proba() can be passed directly to `BenchmarkInfo`.

**Exploring new privacy regimes.** The privacy budget list passed to `BenchmarkInfo` (parameter `eps`) is fully configurable. The user can test a range of values, including high privacy budgets (ε < 0.05).

In summary, `BenchmarkDPFair` is intended to serve as an extensible framework for the research community at the intersection of privacy-preserving machine learning and algorithmic fairness.