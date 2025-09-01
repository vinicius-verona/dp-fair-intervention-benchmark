# Disable TensorFlow warnings
import os
import warnings
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
warnings.filterwarnings("ignore", message=r"(.)*", category=FutureWarning)

import utils.model as model
import utils.dataloader as dtloader
from utils.benchmark import Benchmark
from utils.auxiliar import save_experiment
import pandas as pd
import numpy as np
import gc, os, inspect, types


SEEDS = [] #5,42,253,4112,32645
EPS   = [] #.25, .5, .75, 1, 5, 10, 15, 20
SYNTH = 'aim'


def parse_args():
    parser = argparse.ArgumentParser(description="DP-Fairness Benchmark")
    parser.add_argument(
        "--dataset",
        type=str,
        default="adult",
        choices=["adult", "bank", 'compas', 'bias', 'folktables'],
        help="Dataset name",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="income",
        choices=["income", "y", 'two_year_recid', 'Y', 'PINCP'],
        help="Target column name",
    )
    parser.add_argument(
        "--sensitive_attr",
        type=str,
        default="sex",
        choices=["race", "sex", "age", "marital", "A", "SEX", "RAC1P"],
        help="Sensitive attribute",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[5, 42, 253, 4112, 32645],
        help="List of random seeds, e.g. --seeds 5 42 253 4112 32645",
    )
    parser.add_argument(
        "--combo",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6],
        help="Which bias combo to use, e.g. --combo 1 [ONLY FOR Bias Generated Datasets]",
    )
    parser.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1, 5, 10, 15, 20],
        help="List of ε values for DP, e.g. --eps 0.25 0.5 0.75 1 5 10 15 20",
    )
    
    parser.add_argument(
        "--classifier",
        type=str,
        default='XGB',
        help="Which classifier to use. Available ones are XGB (XGBoost) and LR (LogisticRegression).",
    )
    
    parser.add_argument(
        "--synth",
        type=str,
        default='aim',
        choices=['aim','mst','dpctgan'],
        help="Which differentially private synthesizer to use. Available ones are AIM, MST, DP-CTGAN.",
    )
    

    return parser.parse_args()


def experiment(seed, dataset, sensitive_attr, combo, synth, savefile = "exp_metrics.csv"):
    np.random.seed(seed)

    print(f"\n*********************** No-DP - seed = {seed} ***********************\n")
    dataloader = dtloader.DataLoader()
    file = f"../data/{dataset}/{synth}/DP-dataset-train/split_train_dataset_seed_{seed}.csv"
    
    if dataset == 'Bias' or dataset == 'bias':
        file = f"../data/Biased_datasets/{synth}/Biased_combo_{combo}/DP-dataset-train/split_train_dataset_seed_{seed}.csv"
        
    dataloader.path = file
    dataloader.seed = seed
    original_experiment = Benchmark(name="original", data_loader=dataloader, normalize=True, seed=seed)
    original_experiment.run()

    # save_experiment(original_experiment, seed, filename=savefile, path=f"../data/{dataset}/metrics/")
    if dataset == 'Bias' or dataset == 'bias':
        save_experiment(original_experiment, seed, filename=savefile, path=f"../data/Biased_datasets/{synth}/Biased_combo_{combo}/metrics/",synth=SYNTH)
    else:
        save_experiment(original_experiment, seed, filename=savefile, path=f"../data/{dataset}/{synth}/metrics/",synth=SYNTH)

    del original_experiment, dataloader

    for epsilon in EPS:
        dataloader = dtloader.DataLoader()
        file = f"../data/{dataset}/{synth}/DP-dataset-epsilon-{epsilon}/{synth}_synthetic_train_dataset_seed_{seed}_epsilon_{epsilon}.csv"
                
        if dataset == 'Bias' or dataset == 'bias':
            file = f"../data/Biased_datasets/{synth}/Biased_combo_{combo}/DP-dataset-epsilon-{epsilon}/aim_synthetic_train_dataset_seed_{seed}_epsilon_{epsilon}.csv"
        
        dataloader.path = file
        dataloader.seed = seed

        print(f"\n*********************** DP-aware ε={epsilon} ***********************\n")
        dp_experiment = Benchmark(name=f"dp", data_loader=dataloader, normalize=True, seed=seed)
        dp_experiment.run()

        if dataset == 'Bias' or dataset == 'bias':
            save_experiment(dp_experiment, seed, epsilon, filename=savefile, path=f"../data/Biased_datasets/{synth}/Biased_combo_{combo}/metrics/",synth=SYNTH)
        else:
            save_experiment(dp_experiment, seed, epsilon, filename=savefile, path=f"../data/{dataset}/{synth}/metrics/",synth=SYNTH)

        del dp_experiment.data_loader, dp_experiment, dataloader

def main():
    args           = parse_args()
    dataset        = args.dataset.capitalize()
    target         = args.target
    combo          = args.combo
    classifier     = args.classifier
    synth          = args.synth
    sensitive_attr = args.sensitive_attr
    global SYNTH, SEEDS, EPS
    SYNTH = synth
    
    print(f"Running DP Benchmark on dataset: '{dataset}' with target: '{target}' and sensitive attribute: '{sensitive_attr}'")
    
    dtloader.sensitive_attr = sensitive_attr
    dtloader.target_column  = target
    dtloader.dataset        = dataset
    dtloader.synth          = synth
    model.CLASSIFIER        = classifier
    
    SEEDS = args.seeds
    EPS   = args.eps

    savefile = ""
    if dataset == 'Bias' or dataset == 'bias':
        dtloader.combo = combo
        savefile = f"metrics_seeds_{'_'.join(str(seed) for seed in SEEDS)}_eps_{'_'.join(str(e) for e in EPS)}_Combo_{combo}_synth_{synth}.csv"
    else:
        savefile = f"metrics_seeds_{'_'.join(str(seed) for seed in SEEDS)}_eps_{'_'.join(str(e) for e in EPS)}_synth_{synth}.csv"
    
    for i in range(len(EPS)):
        if EPS[i] == 0:
            EPS = []
        elif EPS[i] >= 1:
            EPS[i] = int(EPS[i])


    for i, seed in enumerate(SEEDS):
        experiment(seed, dataset, sensitive_attr, combo, synth, savefile)

        gc.collect()

        # Memory analysis - Check if all dataframes have been deleted (Issue with iterative calls of AIFF360)
        df_count = sum(1 for obj in gc.get_objects() if isinstance(obj, pd.DataFrame))
        print(f"DataFrames still in memory: {df_count}")

        if df_count > 0:

            for obj in gc.get_objects():
                if isinstance(obj, pd.DataFrame):
                    print(f"\n⚠️ DataFrame id={id(obj)}, shape={obj.shape}")
                    referrers = gc.get_referrers(obj)
                    for ref in referrers:
                        if isinstance(ref, dict):
                            for key, val in ref.items():
                                if val is obj:
                                    print(f"  🔑 Held by dict key: '{key}'")
                                    find_referrers_by_id(obj)
                                    for frame in inspect.stack():
                                        if ref is frame.frame.f_locals:
                                            print(f"    📍 In locals of function: {frame.function} ({frame.filename}:{frame.lineno})")
                                        elif ref is frame.frame.f_globals:
                                            print(f"    📍 In globals of: {frame.function} ({frame.filename}:{frame.lineno})")
                        elif isinstance(ref, types.FrameType):
                            print(f"  ↪️ Referred from frame: {ref.f_code.co_name} ({ref.f_code.co_filename}:{ref.f_lineno})")
                        elif hasattr(ref, '__code__'):
                            print(f"  ↪️ Referred from function: {ref.__name__} ({ref.__code__.co_filename}:{ref.__code__.co_firstlineno})")
                        else:
                            print(f"  ↪️ Referred from: {type(ref)}")


def find_referrers_by_id(target_obj, max_depth=50):
    visited = set()
    target_id = id(target_obj)

    def _search(obj, path='', depth=0):
        if depth > max_depth:
            return
        if id(obj) in visited:
            return
        visited.add(id(obj))

        try:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if id(v) == target_id:
                        print(f"Found in dict at path: {path}[{repr(k)}]")
                    else:
                        _search(v, f'{path}[{repr(k)}]', depth + 1)
            elif isinstance(obj, (list, tuple, set)):
                for i, item in enumerate(obj):
                    if id(item) == target_id:
                        print(f"Found in {type(obj).__name__} at path: {path}[{i}]")
                    else:
                        _search(item, f'{path}[{i}]', depth + 1)
            elif hasattr(obj, '__dict__'):
                for k, v in vars(obj).items():
                    if id(v) == target_id:
                        print(f"Found in object attribute at path: {path}.{k}")
                    else:
                        _search(v, f'{path}.{k}', depth + 1)
            elif isinstance(obj, types.FrameType):
                print(f"  📍 Found frame: {obj.f_code.co_name} ({obj.f_code.co_filename}:{obj.f_lineno})")
                _search(obj.f_locals, f'{path}.f_locals', depth + 1)
                _search(obj.f_globals, f'{path}.f_globals', depth + 1)

        except Exception:
            pass  # Silent fail on inaccessible or problematic objects

    # Start from all referrers of the object
    for ref in gc.get_referrers(target_obj):
        _search(ref, path='gc_root', depth=0)

if __name__ == "__main__":
    main()