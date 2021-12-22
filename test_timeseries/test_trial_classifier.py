import numpy as np
import pytest
import timeseries_data.read_timeseries as rt
import timeseries_data.mean_ts.trial_classifier as trial_class

classification_configurations = [(2, {0:1, 1:0}), (1, {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1}),
                                (1, {0:0, 1:1, 2:1, 3:1}), (1, {0:0, 4:1, 5:1, 6:1}), 
                                (1, {0:0, 1:1}), (1, {0:0, 2:1}),(1, {0:0, 3:1}),(1, {0:0, 4:1}), (1, {0:0, 5:1}),
                                (1, {0:0, 6:1}),([1,2], {(0,1):0, (1,1):1}), ([1,2], {(0,1):0, (2,1):1}),
                                ([1,2], {(0,1):0, (4,1):1}), ([1,2], {(0,1):0, (5,1):1}),([1,2], {(1,0):0, (1,1):1}), 
                                ([1,2], {(2,0):0, (2,1):1}),([1,2], {(4,0):0, (4,1):1}), ([1,2], {(5,0):0, (5,1):1})]
                                     
@pytest.mark.parametrize("num", [2,5,23,27,35,37])
def test_trial_classfier_creator(num):
    data = rt.read_subject(num)
    data = np.array(data)
    
    for config in classification_configurations:
        classifier = trial_class.trial_classfier_creator(idx=config[0], class_dict=config[1])
        
        for line in data:
            class_idx = classifier(line)
            
            key = line[config[0]] if isinstance(config[0], int) else tuple(line[config[0]])
            real_idx = config[1].get(key, 2) 
            
            assert real_idx == class_idx
            