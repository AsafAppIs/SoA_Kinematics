import numpy as np
import pytest
import timeseries_data.read_timeseries as rt
import timeseries_data.mean_ts.trial_classifier as trial_class
import timeseries_data.mean_ts.subject_classifier as sub_class
import timeseries_data.mean_ts.ts_statistics as ts_stat
classification_configurations = [(2, {0:1, 1:0}), (1, {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1}),
                                (1, {0:0, 1:1, 2:1, 3:1}), (1, {0:0, 4:1, 5:1, 6:1}), 
                                (1, {0:0, 1:1}), (1, {0:0, 2:1}),(1, {0:0, 3:1}),(1, {0:0, 4:1}), (1, {0:0, 5:1}),
                                (1, {0:0, 6:1}),([1,2], {(0,1):0, (1,1):1}), ([1,2], {(0,1):0, (2,1):1}),
                                ([1,2], {(0,1):0, (4,1):1}), ([1,2], {(0,1):0, (5,1):1}),([1,2], {(1,0):0, (1,1):1}), 
                                ([1,2], {(2,0):0, (2,1):1}),([1,2], {(4,0):0, (4,1):1}), ([1,2], {(5,0):0, (5,1):1})]
   
@pytest.mark.parametrize("num", [2,5,23,27,35,37])                                  
def test_subject_classifier(num):
    data = rt.read_subject(num)
    data = np.array(data)
    
    for config in classification_configurations:
        classifier = trial_class.trial_classfier_creator(idx=config[0], class_dict=config[1])
        
        idx1, idx2 = sub_class.subject_classifier(data, classifier)
        
        for line in data[idx1]:            
            key = line[config[0]] if isinstance(config[0], int) else tuple(line[config[0]])
            real_idx = config[1].get(key, 2) 
            
            assert real_idx == 0
            
            
        for line in data[idx2]:            
            key = line[config[0]] if isinstance(config[0], int) else tuple(line[config[0]])
            real_idx = config[1].get(key, 2) 
            
            assert real_idx == 1
            
            
def test_conditional_mean_ts():
    data = rt.read_subject(100)
    
    results = ((10,3), (3.5,7), (3.5,5.5), (3.5,8.5), (7,8))
    for i, config in enumerate(classification_configurations[:4] + classification_configurations[10:11]):
        first_class_mean, _, second_class_mean, _ = sub_class.conditional_mean_ts(data, idx=config[0], class_dict=config[1])
        
        assert (first_class_mean == results[i][0]).all()
        assert (second_class_mean == results[i][1]).all()