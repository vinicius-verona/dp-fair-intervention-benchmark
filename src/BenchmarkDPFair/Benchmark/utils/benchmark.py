EXP_CLASSES = ["original", "pre", "pos", "in"]

# from memory_profiler import profile
import sys
import traceback
import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from xgboost import XGBClassifier

# from .dataloader import sensitive_attr
from .pre import pre_mitigator_experiment
from .inp import in_mitigator_experiment
from .pos import pos_mitigator_experiment

from .auxiliar import getMetrics
from .verifiers import check_signatures

import gc

def original_experiment(x_train, y_train, x_test, y_test, sensitive_attr, target_column, seed=42, normalize=True, threshold=.5):
    # import utils.dataloader as dataloader
    # sensitive_attr, target_column, dataset = dataloader.sensitive_attr, dataloader.target_column, dataloader.dataset

    

    # if dataset is COMPAS, switch
    privileged_groups = [{sensitive_attr: 1}] # Ex: White
    unprivileged_groups = [{sensitive_attr: 0}] # Ex: Not white
    
    scaler = None
    if normalize:
        scaler = MinMaxScaler()

    if scaler is not None:
        cols = x_train.columns
        x_train = scaler.fit_transform(x_train)
        x_train = pd.DataFrame(x_train, columns=cols)
        x_test = scaler.transform(x_test)
        x_test = pd.DataFrame(x_test, columns=cols)
        
        
    model = XGBClassifier(objective='binary:logistic', random_state=seed)
    # model = LogisticRegression(**ESTIMATOR_PARAMS)
    model.fit(x_train, y_train)
    y_pred_prob = model.predict_proba(x_test)[:, 1]
    y_pred = (y_pred_prob >= threshold).astype(int)
        
    y_preds = pd.DataFrame(y_pred, columns=[target_column])
    
    # Reset the index
    y_preds = y_preds.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    og_dataset_test = pd.concat([x_test, y_preds], axis=1)
        
    
    og_dataset_test_pred = BinaryLabelDataset(df=og_dataset_test, label_names=[target_column], protected_attribute_names=[sensitive_attr], 
                                unprivileged_protected_attributes=unprivileged_groups)


    y_test = y_test.reset_index(drop=True)
    df_test = pd.concat([x_test, y_test], axis=1)
    df_test = BinaryLabelDataset(df=df_test, label_names=[target_column], protected_attribute_names=[sensitive_attr], 
                                unprivileged_protected_attributes=unprivileged_groups)


    og_classification_metrics = ClassificationMetric(df_test, og_dataset_test_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    og_metrics = getMetrics(og_classification_metrics)
    
    og_classification_metrics.dataset            = None
    og_classification_metrics.classified_dataset = None

    del scaler,  y_preds, model, x_train, y_train, x_test, y_test, og_dataset_test, y_pred, y_pred_prob, og_dataset_test_pred, df_test, og_classification_metrics

    return {
        "original_classification_metrics": og_metrics,
    }


class Benchmark:
    def __init__(self, name, data_loader, normalize=None, seed=42, verbose=False, threshold=.5, dlkwargs=None, ekwargs = None):
        """
        :param name: name of the experiment
        :param model: instance of a ML model.
        :param normalize: should use a normalizer
        :param data_loader: instance of DataLoader
        :param seed: seed for reproducibility
        :param verbose: verbosity of the experiment
        """
        self.name        = name
        self.data_loader = data_loader
        self.normalize   = normalize
        self.seed        = seed
        self.verbose     = verbose
        self.mitigators  = {
            "reweigh":"pre",
            "dir": "pre",
            "lfr": "pre",
            "egr": "in", 
            "gsr": "in",
            "roc":"pos",
            "eqodds":"pos",
            "ceop": "pos",
        }
        self.threshold   = threshold
        self.results = []
        self.dlkwargs = dlkwargs
        self.ekwargs = ekwargs
        
        
    def run(self):
        
        args = check_signatures(self.data_loader, self.dlkwargs|self.ekwargs)
        train_data, cal_data, test_data = self.data_loader(**args)
        # if not self.ekwargs.custom_loader:
        #     train_data, cal_data, test_data = self.data_loader(self.ekwargs.data_conf, self.ekwargs.filename, self.seed, self.ekwargs.eps, **self.dlkwargs)
        # else:
        #     train_data, cal_data, test_data = self.data_loader(seed=self.seed,**self.dlkwargs)

        
        X_train, y_train = train_data[0].copy(), train_data[1].copy()
        X_cal, y_cal     = cal_data[0].copy(), cal_data[1].copy()
        X_test, y_test   = test_data[0].copy(), test_data[1].copy()
        
        
        # Run the original experiment
        print("# Original - ", end="")
        try:
            self.results.append(original_experiment(X_train, y_train, X_test, y_test, self.dlkwargs.data_conf.sensitive_attr, self.dlkwargs.data_conf.target, self.seed, self.normalize, self.threshold))
        except Exception as e:
            self.results.append({"original_classification_metrics": getMetrics(None), "error": e})
        print("OK", flush=True)

        
        # Run the experiment with mitigators 
        for mitigator, exp_class in self.mitigators.items():
            print(f"# {exp_class.upper()} - {mitigator.upper()} - ", end="")
            
            X_train, y_train = train_data[0].copy(), train_data[1].copy()
            X_cal, y_cal     = cal_data[0].copy(), cal_data[1].copy()
            X_test, y_test   = test_data[0].copy(), test_data[1].copy()

            try: 
                if exp_class == "pre":
                    self.results.append(pre_mitigator_experiment(X_train, y_train, X_cal, y_cal, X_test, y_test, mitigator, self.seed, self.normalize, self.threshold))
                elif exp_class == "pos":
                    self.results.append(pos_mitigator_experiment(X_train, y_train, X_cal, y_cal, X_test, y_test, mitigator, self.seed, self.normalize, self.threshold))
                else:
                    self.results.append(in_mitigator_experiment(X_train, y_train, X_cal, y_cal, X_test, y_test, mitigator, self.seed, self.normalize, self.threshold))
            
            except Exception as e:
                self.results.append({"mitigator": mitigator, "original_classification_metrics": getMetrics(None), "mitigated_classification_metrics": getMetrics(None), "error": e, "dp_method": True, 'info': traceback.format_tb(e.__traceback__)})#sys.exc_info()[-1].tb_lineno})

            X_train = None
            y_train = None
            X_cal   = None
            y_cal   = None
            X_test  = None
            y_test  = None
            
            del X_train, y_train, X_cal, y_cal, X_test, y_test
            print("OK", flush=True)

        del train_data, cal_data, test_data
        del dataset
        gc.collect()

            
