# IO Pattern of DataGeneration

This benchmark expects to find the following structural pattern for csv storage **Be aware, all strings are case sensitive**

An example will be given by the end. 
Step 4 will be automatically generated with our `generate_data()`.

1. Within the set `root_dir` (data generation) argument, there **must** exist a directory with the name set in `DatasetGeneratorConfig`.
2. Within this directory, it expects another with the name chosen on `name` argument of `DatasetGeneratorConfig`.
3. Lastly, within the `root_dir/name` (`./data/Compas/` in the example below), there must exists the file provided to `generate_data()`
4. In such directory, we expect to find a directory with the synthesizer name and inside, three types of subdirectories:
    * DP-dataset-train, with the original training and calibration data
    * DP-dataset-test, with the original test data
    * DP-dataset-epsilon-[X], with the synthetic data for epsilon X

If the original csv, containing all original data, is not within a directory following pattern 1, please use the path argument in `generate_data()` to specify the correct path to search for the csv.

## Example
Here is a simple exmaple based on [`./example/dummy.py`](./example/dummy.py) script.
```py
"""
./Project
    |- example/
        |- DPF_DataGeneration.py
        |- data/
            |- Compas/
                |- compas.csv
                |- aim **[Generated]**
                    |- DP-dataset-train/        **[Generated]**
                    |- DP-dataset-test-val/     **[Generated]**
                    |- DP-dataset-epsilon-0.25/ **[Generated]**
"""

synhts = ["aim", "mst"]
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

            generate_data(f"compas.csv", "", data_conf, "./data/Compas/", verbose=True)
```