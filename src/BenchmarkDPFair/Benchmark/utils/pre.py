from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset
from IPython.display import Markdown, display
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from aif360.metrics import ClassificationMetric

from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing, LFR, OptimPreproc, DisparateImpactRemover
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import get_distortion_adult


from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .auxiliar import getMetrics
from xgboost import XGBClassifier

import numpy as np
import pandas as pd

optim_options = {
    # "distortion_fun": get_distortion_adult,
    "epsilon": 0.05,
    "clist": [0.99, 1.99, 2.99],
    "dlist": [.1, 0.05, 0]
}

def pre_mitigator_experiment(X_train, y_train, X_cal, y_cal, X_test, y_test, mitigator, seed=42, normalize=True, threshold=.5):
    
    import utils.dataloader as dataloader
    sensitive_attr, target_column, dataset = dataloader.sensitive_attr, dataloader.target_column, dataloader.dataset
    
    # if dataset is COMPAS, switch
    privileged_groups = [{sensitive_attr: 1}] # Ex: White
    unprivileged_groups = [{sensitive_attr: 0}] # Ex: Not white
    
    if mitigator not in ["lfr", "reweigh", "optp", "dir", "reweighin", "DisparateImpactRemover", "optim_preproc"]:
        raise ValueError(f"Invalid mitigator {mitigator}. Choose between 'lfr', 'reweigh', 'optp' and 'dir'")

    
    if normalize:
        scaler = MinMaxScaler()
    
    # Convert to BinaryLabelDataset
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
    cal_set = X_cal.reset_index(drop=True)
    target_cal = y_cal.reset_index(drop=True)
    df_cal = pd.concat([cal_set, target_cal], axis=1)
    df_cal = BinaryLabelDataset(df=df_cal, label_names=[target_column],
                                protected_attribute_names=[sensitive_attr], 
                                unprivileged_protected_attributes=unprivileged_groups)

    # Test
    test_set = X_test.reset_index(drop=True)
    target_test = y_test.reset_index(drop=True)
    
    if normalize:
        cols = test_set.columns
        test_set = scaler.transform(test_set)
        test_set = pd.DataFrame(test_set, columns=cols)
        
    df_test = pd.concat([test_set, target_test], axis=1)
    df_test = BinaryLabelDataset(df=df_test, label_names=[target_column],
                                 protected_attribute_names=[sensitive_attr], 
                                 unprivileged_protected_attributes=unprivileged_groups)

    # ####################################################################
    # ####################################################################
    # ########################## Model Training ##########################
    # ####################################################################
    # ####################################################################
    
    classifier = XGBClassifier(objective='binary:logistic', random_state=seed)#LogisticRegression(**ESTIMATOR_PARAMS, random_state=seed)
    # classifier = LogisticRegression(**ESTIMATOR_PARAMS, random_state=seed)
    classifier.fit(train_set, target.to_numpy())
    y_pred_prob = classifier.predict_proba(test_set)[:, 1]
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    og_test_preds_dataset = df_test.copy()
    og_test_preds_dataset.labels = y_pred
    og_test_preds_dataset.scores = y_pred_prob

    ###############################################################
    ########################## Mitigator ##########################
    ###############################################################
    
    mechanism = None
    
    if mitigator == "reweigh" or mitigator == "reweighing":
        mechanism = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    elif mitigator == "lfr":
        mechanism = LFR(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups, seed=seed)
    elif mitigator == "optim_preproc" or mitigator == "optp":
        mechanism = OptimPreproc(OptTools, optim_options, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups, seed=seed)
    elif mitigator == "dir" or mitigator == "DisparateImpactRemover":
        mechanism = DisparateImpactRemover(repair_level=1, sensitive_attribute=sensitive_attr)
        
    if mechanism is None:
        raise ValueError("Mitigator not found")

    # print("Before:", np.unique(df_train.labels, return_counts=True))
    if mitigator != "dir" and mitigator != "DisparateImpactRemover":
        # Fit mitigator on training data
        try:
            mechanism = mechanism.fit(df_train)
        except Exception as e:
            del mechanism, classifier, test_set, df_test, target_test, scaler, train_set, target, df_train, df_cal, target_cal, cal_set
            raise e
        try:
            mitigated_train = mechanism.transform(df_train)
        except Exception as e:
            del mitigated_cal, mechanism, classifier, test_set, df_test, target_test, scaler, train_set, target, df_train, df_cal, target_cal, cal_set, mitigated_train
            raise e
        
        try:
            mitigated_cal   = mechanism.transform(df_cal)
        except Exception as e:
            del mitigated_cal, mechanism, classifier, test_set, df_test, target_test, scaler, train_set, target, df_train, df_cal, target_cal, cal_set, mitigated_train
            raise e
    else:
        # Transform ALL datasets needed using the fitted mitigator
        mitigated_train = mechanism.fit_transform(df_train)
        mitigated_cal   = mechanism.fit_transform(df_cal)

    # print("After:", np.unique(mitigated_train.labels, return_counts=True))
    mitigated_test = df_test.copy(deepcopy=True)
    
    try:
        if mitigator == "LFR" or mitigator == "lfr":
            mitigated_test = mechanism.transform(df_test) # only on LFR
            
        elif mitigator == "dir" or mitigator == "DisparateImpactRemover":
            mitigated_test = mechanism.fit_transform(df_test) 
    except Exception as e:
        del mitigated_cal, mechanism, classifier, test_set, df_test, target_test, scaler, train_set, target, df_train, df_cal, target_cal, cal_set, mitigated_train
        del mitigated_test
        raise e
        
    # Calculate weights for each dataset
    mitigated_train_weights = mitigated_train.instance_weights

    ############################################################################
    ################## Retrain Classifier on Transformed Data ##################
    ############################################################################

    # Extract features and labels from transformed data
    X_train_transf = pd.DataFrame(mitigated_train.features, columns=X_train.columns)
    y_train_transf = mitigated_train.labels

    try:
        mitigator_model = XGBClassifier(objective='binary:logistic', random_state=seed)#LogisticRegression(**ESTIMATOR_PARAMS, random_state=seed)   
        # mitigator_model = LogisticRegression(**ESTIMATOR_PARAMS, random_state=seed)   
        mitigator_model.fit(X_train_transf, y_train_transf, sample_weight=mitigated_train_weights)
    
    except Exception as e:
        del mitigated_cal, mechanism, classifier, test_set, df_test, target_test, scaler, train_set, target, df_train, df_cal, target_cal, cal_set, mitigated_train
        del mitigated_test, X_train_transf, mitigated_train_weights, mitigator_model, y_train_transf
        raise e
    
    ###############################################################################
    ###################### Evaluate on Transformed Test Data ######################
    ###############################################################################

    try:
        # Predict using transformed features
        y_pred_prob_mit = mitigator_model.predict_proba(pd.DataFrame(mitigated_test.features, columns=mitigated_test.feature_names))[:, 1]
    
    except Exception as e:
        del mitigated_cal, mechanism, classifier, test_set, df_test, target_test, scaler, train_set, target, df_train, df_cal, target_cal, cal_set, mitigated_train
        del mitigated_test, X_train_transf, mitigated_train_weights, mitigator_model, y_pred_prob_mit
        raise e
    
    y_pred_mit = (y_pred_prob_mit >= threshold).astype(int)
    
    mitigator_test_preds = df_test.copy(deepcopy=True) #mitigated_test.copy()
    mitigator_test_preds.labels = y_pred_mit#.reshape(-1, 1)
    mitigator_test_preds.scores = y_pred_prob_mit

    #############################################################
    ########################## Metrics ##########################
    #############################################################

    # Original model metrics
    og_classification_metrics = ClassificationMetric(
        df_test, 
        og_test_preds_dataset,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )

    # Mitigated model metrics
    pre_classification_metrics = ClassificationMetric(
        df_test, 
        mitigator_test_preds,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    og_metrics = getMetrics(og_classification_metrics)
    fair_metrics = getMetrics(pre_classification_metrics)

    og_classification_metrics.dataset             = None
    og_classification_metrics.classified_dataset  = None
    pre_classification_metrics.dataset            = None
    pre_classification_metrics.classified_dataset = None

    del og_classification_metrics, pre_classification_metrics, mitigator_test_preds, y_pred_mit, y_pred_prob_mit, mitigator_model, y_train_transf, X_train_transf, mitigated_train_weights
    del mitigated_test, mitigated_cal, mechanism, classifier, test_set, df_test, target_test, scaler, train_set, target, df_train, df_cal, target_cal, cal_set

    return {
        "original_classification_metrics": og_metrics,
        "mitigated_classification_metrics": fair_metrics,
        "mitigator": mitigator,
        "dp_method": True
    }
    
