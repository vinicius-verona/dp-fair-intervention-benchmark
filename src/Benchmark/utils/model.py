import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

CLASSIFIER = 'XGB' # or LR (LogisticRegression)
ESTIMATOR_PARAMS = {
    'max_iter': 10000,
    'solver': 'saga',
    'penalty': 'elasticnet',
    'l1_ratio': 0.5,
}
