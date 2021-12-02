import numpy as np
import pandas as pd
import pytest
import timeseries_data.configurations as cfg
from timeseries_data.roc_curve.roc import auc_calculator
from timeseries_data.mean_ts.subject_classifier import subject_classifier
from timeseries_data.mean_ts.trial_classifier import trial_classfier_creator

@pytest.mark.parametrize("num, result", [(1,1),(2,.5)])
def test_roc_curve(num, result):
    df = pd.read_csv(cfg.special_feature_path + "fake" + str(num) + ".csv", header=None)
    data = np.array(df)
    config = "soa"
    # define classifier function
    classifier = trial_classfier_creator(config=config)
    # get indices of both classes
    first_class_idx, second_class_idx = subject_classifier(data, classifier)
    
    assert auc_calculator(data, first_class_idx, second_class_idx, 0) == result