# IO Pattern

In its first version, this benchmark expects to find the following structural pattern for csv storage **Be aware, all strings are case sensitive**

Examples will be given by the end. All steps from 2 onwards will be automatically generated with our `generate_data()`.

1. Within the set `root_dir` (data generation) there **must** exist a directory with the name set in `DatasetGeneratorConfig`.
2. Within this directory, it expects another with the name set to the synthesizer chosen.
3. In such directory, we expect three types of subdirectories:
    * DP-dataset-train, with the original training and calibration data
    * DP-dataset-test-val, with the original test data
    * DP-dataset-epsilon-[X], with the synthetic data for epsilon X

If the original csv, containing all original data, is not within a directory following pattern 1, please use the path argument in `generate_data()` to specify the correct path to search for the csv.

## Example
Here is a simple exmaple used on [`DPF_DataGeneration.py`](./example/DPF_DataGeneration.py) script.
```py
"""
./Project
    |- example/
        |- DPF_DataGeneration.py
        |- data/
            |- Adult/
                |- adult.csv
                |- aim **[Generated]**
                    |- DP-dataset-train/        **[Generated]**
                    |- DP-dataset-test-val/     **[Generated]**
                    |- DP-dataset-epsilon-0.25/ **[Generated]**
"""

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

generate_data("adult.csv", data_conf, verbose=True) # Since the directory follows the pattern, no need to set path
```