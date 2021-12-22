import numpy as np
import pandas as pd
import pytest
import timeseries_data.configurations as cfg
from timeseries_data.roc_curve.roc import auc_calculator, subject_roc_curve
import timeseries_data.roc_curve.auc_calculation as auc_cal
from timeseries_data.roc_curve.special_feature_translation import to_special_features
from timeseries_data.mean_ts.subject_classifier import subject_classifier
from timeseries_data.mean_ts.trial_classifier import trial_classfier_creator
from timeseries_data.read_timeseries import read_subject

def test_special_features_translation():
    data = read_subject(1,True)
    data = np.array(data)
    new_representation = to_special_features(data)
    data = new_representation
    
    assert data[0,3] == data[0,4] == data[0,5] == 0
    assert data[0,6] == 30
    assert data[0,7] == 30

    assert data[20,3] == data[20,4] == data[20,5] == -1
    assert data[20,6] == 40
    assert data[20,7] == 30


    assert data[40,3] == data[40,5] == data[40,5] == -2
    assert data[40,6] == 50
    assert data[40,7] == 30

def test_subject_auc():
    results = auc_cal.subject_auc(100)
    
    assert results[1] == results[2] == results[3] == results[4] == results[5] == 1
    assert results[6] == results[7] == results[8] == results[9] == results[10] == pytest.approx(.75)
    assert results[11] == results[12] == results[13] == results[14] == results[15] == pytest.approx(.75)
    assert results[21] == results[22] == results[23] == results[24] == results[25] == .625
    assert results[31] == results[32] == results[33] == results[34] == results[35] == .875
    
    
def test_subject_roc_curve():
    data = pd.read_csv(cfg.special_feature_path + "participant" + str(100) + ".csv", header=None)
    data = np.array(data)
    fp, tp = subject_roc_curve(subject_data=data, measure_idx=1, config="manipulation_s")
    
    assert (fp == [0,.25,1]).all()
    assert (tp == [0,.75,1]).all()
    
test_subject_roc_curve()

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
    
    