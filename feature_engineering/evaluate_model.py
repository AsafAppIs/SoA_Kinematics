import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, LeavePOut
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score

import feature_engineering.configurations as cfg
from augmentate import create_synthetic_data, upsample_data
from feature_engineering.data_preperation import prepre_data
from feature_engineering.utils.unconfound import soa_unconfound

def evaluate(X, Y, model, validation_method, test_mode=False):
    # define validation method 
    if validation_method == "cv":
        kf = StratifiedKFold(n_splits=cfg.k_validation, shuffle=True, random_state=cfg.random_seed)
    elif validation_method == "lto":
        kf = LeavePOut(2)
    
    results = []
    total_true = []
    total_score = []
    
    # iterate over the folds
    for train_index, test_index in kf.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        
        
        # create synthetic data
        #x_train, y_train = create_synthetic_data(x_train, y_train, k=3)
        x_train, y_train = upsample_data(x_train, y_train)
        
        
        # standartize data
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
        
        # fit model        
        model.fit(x_train, y_train)
        
        
        # calculate confusion matrix
        y_hat = model.predict(x_test)
        results.append(confusion_matrix(y_test, y_hat))
        
        # calculate probablity
        y_hat = model.predict_proba(x_test)
        y_prob = list(y_hat[:,1])
        
        
        
        # add y and y_hat to results lists
        total_true += list(y_test)
        total_score += y_prob
        
        
    # sum up confusion matrices
    #conf = sum(results)
    # calculate auc
    auc = roc_auc_score(total_true, total_score)   
    
    # calculate confusion matrix
    confusion = sum(results)
    
    
    # in testing mode we will want to check whether the randomality is static (reproductability&static folds)
    if test_mode:
        return auc, kf.split(X)
    
    return auc, confusion



