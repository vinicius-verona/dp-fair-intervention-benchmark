# In-processing Fairness for the DP-Benchmark
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.inprocessing import ExponentiatedGradientReduction, GridSearchReduction
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from aif360.metrics import ClassificationMetric
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

from .auxiliar import getMetrics



#############################################################
#############################################################
########################## Dataset ##########################
#############################################################
#############################################################
def in_mitigator_experiment(X_train, y_train, X_cal, y_cal, X_test, y_test, sensitive_attr, target_column, mitigator, seed=42, normalize=True, threshold=.5):
    
    # import utils.dataloader as dataloader
    # sensitive_attr, target_column, dataset = dataloader.sensitive_attr, dataloader.target_column, dataloader.dataset
    
    # if dataset is COMPAS, switch
    privileged_groups = [{sensitive_attr: 1}] # Ex: White
    unprivileged_groups = [{sensitive_attr: 0}] # Ex: Not white
    
    if mitigator not in ["egr", "gsr"]:
        raise ValueError(f"Invalid mitigator {mitigator}. Choose between 'egr' and 'gsr'")
    
    
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

    # Calibrate
    calibration_set = X_cal.reset_index(drop=True)
    target_cal = y_cal.reset_index(drop=True)
    df_cal = pd.concat([calibration_set, target_cal], axis=1)
    df_cal = BinaryLabelDataset(df=df_cal, label_names=[target_column], 
                                protected_attribute_names=[sensitive_attr], 
                                unprivileged_protected_attributes=unprivileged_groups)
    
    # Test
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
    og_model = XGBClassifier(objective='binary:logistic', random_state=seed)#LogisticRegression(**ESTIMATOR_PARAMS)
    mitigator_model = XGBClassifier(objective='binary:logistic', random_state=seed)#LogisticRegression(**ESTIMATOR_PARAMS)
    # og_model = LogisticRegression(**ESTIMATOR_PARAMS)
    # mitigator_model = LogisticRegression(**ESTIMATOR_PARAMS)
    
    # Test set
    dataset_orig_test = df_test.copy(deepcopy=True)
    dataset_mitigated_test = df_test.copy(deepcopy=True)
    
    # Fit mitigator using calibration set
    if mitigator == "egr":
        in_mitigator = ExponentiatedGradientReduction(mitigator_model, "EqualizedOdds", drop_prot_attr=False)
    elif mitigator == "gsr":
        in_mitigator = GridSearchReduction(mitigator_model, "EqualizedOdds", drop_prot_attr=False, prot_attr=sensitive_attr)
        # in_mitigator = GridSearchReduction(mitigator_model, "EqualizedOdds", drop_prot_attr=False, prot_attr=sensitive_attr, grid_size=100)
        

    # Fit using calibration set (true labels in df_cal, predictions in dataset_orig_cal_pred)
    og_model = og_model.fit(train_set, target.to_numpy())
    in_mitigator = in_mitigator.fit(df_train)

    # Apply mitigation to test set
    og_model_dataset_predictions = og_model.predict(test_set)
    dataset_orig_test.labels = og_model_dataset_predictions
    
    in_mitigated_dataset_predictions = in_mitigator.predict(dataset_mitigated_test)
    in_mitigated_dataset_predictions.labels = in_mitigated_dataset_predictions.labels.astype(int)

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

    in_classification_metrics = ClassificationMetric(
        df_test,  # True labels
        in_mitigated_dataset_predictions,  # Mitigated predictions
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )

    og_metrics = getMetrics(og_classification_metrics)
    fair_metrics = getMetrics(in_classification_metrics)

    og_classification_metrics.dataset            = None
    og_classification_metrics.classified_dataset = None
    in_classification_metrics.dataset            = None
    in_classification_metrics.classified_dataset = None


    del scaler,train_set,target,df_train,calibration_set,target_cal,df_cal,test_set,target_test,df_test,
    del og_model,mitigator_model,dataset_orig_test,dataset_mitigated_test,in_mitigator,og_model_dataset_predictions,
    del in_mitigated_dataset_predictions,og_classification_metrics,in_classification_metrics

    return {
        "original_classification_metrics": og_metrics,
        "mitigated_classification_metrics": fair_metrics,
        "mitigator": mitigator,
        "dp_method": True
    }


