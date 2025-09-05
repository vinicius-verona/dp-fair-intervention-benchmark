from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset, RegressionDataset
# from IPython.display import Markdown, display
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, EqOddsPostprocessing, RejectOptionClassification, DeterministicReranking
from sklearn.preprocessing import MinMaxScaler
from aif360.metrics import ClassificationMetric
from xgboost import XGBClassifier

from .auxiliar import getMetrics

import numpy as np
import pandas as pd

# privileged_groups = [{sensitive_attr: 1}] # White
# unprivileged_groups = [{sensitive_attr: 0}] # Not white

#############################################################
#############################################################
########################## Dataset ##########################
#############################################################
#############################################################

def pos_mitigator_experiment(X_train, y_train, X_cal, y_cal, X_test, y_test, sensitive_attr, target_column, mitigator, seed=42, normalize=True, threshold=.5):
    
    # import utils.dataloader as dataloader
    # sensitive_attr, target_column, categorical_cols, sensitive_columns = dataloader.sensitive_attr, dataloader.target_column, dataloader.categorical_cols, dataloader.sensitive_columns
    
    # if dataset is COMPAS, switch
    privileged_groups = [{sensitive_attr: 1}] # Ex: White
    unprivileged_groups = [{sensitive_attr: 0}] # Ex: Not white
    
    if mitigator not in ["eqodds", "roc", "ceop"]:
        raise ValueError(f"Invalid mitigator {mitigator}. Choose between 'eqodds' and 'roc'")
    
    scaler = None
    if normalize:
        scaler = MinMaxScaler()
        
    train_set = X_train.reset_index(drop=True)
    target = y_train.reset_index(drop=True)
    
    if scaler is not None:
        cols = train_set.columns
        train_set = scaler.fit_transform(train_set)
        train_set = pd.DataFrame(train_set, columns=cols)

    df_train = pd.concat([train_set, target], axis=1)
    df_train = BinaryLabelDataset(df=df_train, label_names=[target_column], 
                                  protected_attribute_names=[sensitive_attr], 
                                    unprivileged_protected_attributes=unprivileged_groups)

    calibration_set = X_cal.reset_index(drop=True)
    target_cal = y_cal.reset_index(drop=True)
    
    if scaler is not None:
        cols = calibration_set.columns
        calibration_set = scaler.transform(calibration_set)
        calibration_set = pd.DataFrame(calibration_set, columns=cols)

    df_cal = pd.concat([calibration_set, target_cal], axis=1)
    df_cal = BinaryLabelDataset(df=df_cal, label_names=[target_column],
                                protected_attribute_names=[sensitive_attr], 
                                unprivileged_protected_attributes=unprivileged_groups)
    
    test_set = X_test.reset_index(drop=True)
    target_test = y_test.reset_index(drop=True)
    
    if scaler is not None:
        cols = test_set.columns
        test_set = scaler.transform(test_set)
        test_set = pd.DataFrame(test_set, columns=cols)

    df_test = pd.concat([test_set, target_test], axis=1)
    df_test = BinaryLabelDataset(df=df_test, label_names=[target_column],
                                 protected_attribute_names=[sensitive_attr], 
                                 unprivileged_protected_attributes=unprivileged_groups)


    ###################################################################
    ###################################################################
    ######################### Model Training ##########################
    ###################################################################
    ###################################################################


    # scaler = MinMaxScaler()
    og_model = XGBClassifier(objective='binary:logistic', random_state=seed)#LogisticRegression(**ESTIMATOR_PARAMS)
    # og_model = LogisticRegression(**ESTIMATOR_PARAMS)
    og_model.fit(train_set, target.to_numpy())
    
    
    ##############################################################
    ######################### Mitigator ##########################
    ##############################################################

    # Generate predictions for calibration and test sets
    y_cal_pred_prob = og_model.predict_proba(calibration_set)[:, 1] 
    y_pred_cal = (y_cal_pred_prob >= threshold).astype(int)
    
    y_test_pred_prob = og_model.predict_proba(test_set)[:, 1] 
    y_pred = (y_test_pred_prob >= threshold).astype(int)

    # For calibration set
    dataset_orig_cal_pred = df_cal.copy(deepcopy=True)
    dataset_orig_cal_pred.scores = y_cal_pred_prob.reshape(-1, 1)
    dataset_orig_cal_pred.labels = y_pred_cal.reshape(-1, 1)

    # Test set
    dataset_orig_test = df_test.copy(deepcopy=True)
    dataset_orig_test.scores = y_test_pred_prob.reshape(-1, 1)
    dataset_orig_test.labels = y_pred.reshape(-1, 1)

    # Fit mitigator using calibration set
    if mitigator == "eqodds":
        post_mitigator = EqOddsPostprocessing(privileged_groups=privileged_groups, 
                                                  unprivileged_groups=unprivileged_groups,
                                                  seed=seed)
    elif mitigator == "roc":
        post_mitigator = RejectOptionClassification(privileged_groups=privileged_groups,
                                                        unprivileged_groups=unprivileged_groups)
        
    elif mitigator == "ceop":
        post_mitigator = CalibratedEqOddsPostprocessing(privileged_groups=privileged_groups,
                                                        unprivileged_groups=unprivileged_groups,
                                                        seed=seed, cost_constraint="fnr")

    try:
        # Fit using calibration set (true labels in df_cal, predictions in dataset_orig_cal_pred)
        post_mitigator = post_mitigator.fit(df_cal, dataset_orig_cal_pred)
    except Exception as e:
        del scaler, train_set, target, df_train, calibration_set, target_cal, df_cal, test_set
        del target_test, df_test, og_model, y_cal_pred_prob, y_pred_cal, y_test_pred_prob, y_pred,
        del dataset_orig_cal_pred, dataset_orig_test, post_mitigator

        raise e

    try:
        # Apply mitigation to test set
        post_mitigated_dataset = post_mitigator.predict(dataset_orig_test)

    except Exception as e:
        del scaler, train_set, target, df_train, calibration_set, target_cal, df_cal, test_set
        del target_test, df_test, og_model, y_cal_pred_prob, y_pred_cal, y_test_pred_prob, y_pred,
        del dataset_orig_cal_pred, dataset_orig_test, post_mitigator
        raise e
    ##############################################################
    ######################### Metrics ############################
    ##############################################################

    # Use ClassificationMetric for all evaluations
    og_classification_metrics = ClassificationMetric(
        df_test,  # True labels
        dataset_orig_test,  # Original predictions
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )

    pos_classification_metrics = ClassificationMetric(
        df_test,  # True labels
        post_mitigated_dataset,  # Mitigated predictions
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )

    og_metrics   = getMetrics(og_classification_metrics)
    fair_metrics = getMetrics(pos_classification_metrics)

    og_classification_metrics.dataset             = None
    og_classification_metrics.classified_dataset  = None
    pos_classification_metrics.dataset            = None
    pos_classification_metrics.classified_dataset = None

    del scaler, train_set, target, df_train, calibration_set, target_cal, df_cal, test_set
    del target_test, df_test, og_model, y_cal_pred_prob, y_pred_cal, y_test_pred_prob, y_pred,
    del dataset_orig_cal_pred, dataset_orig_test, post_mitigator, post_mitigated_dataset, og_classification_metrics, pos_classification_metrics
    return {
        "original_classification_metrics": og_metrics,
        "mitigated_classification_metrics": fair_metrics,
        "mitigator": mitigator,
        "dp_method": True
    }

