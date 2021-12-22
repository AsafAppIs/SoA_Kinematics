import pytest
import numpy as np
import timeseries_data.read_timeseries as rt
import timeseries_data.configurations as cfg
import timeseries_data.util.util as util
import timeseries_data.util.soa_equal as equal
from timeseries_data.mean_ts.trial_classifier import trial_classfier_creator
from timeseries_data.mean_ts.subject_classifier import subject_classifier


@pytest.mark.parametrize("num", [1,6,13,24,31,40])
def test_erase_and_count(num):
    data = rt.read_subject(num)
    data = np.array(data)
    
    # split the data according to soa answer
    classifier = trial_classfier_creator(config="soa")
    idx1, idx0 = subject_classifier(data, classifier)
    
    for maniplulation_type in range(7):
        new_idx1 = equal.erase_trials(data, maniplulation_type, 1, idx1)
        trial_counts = equal.trial_type_counter(data[new_idx1])       
        assert trial_counts[maniplulation_type] == 0
        
        new_idx0 = equal.erase_trials(data, maniplulation_type, 0, idx0)
        trial_counts = equal.trial_type_counter(data[new_idx0])       
        assert trial_counts[maniplulation_type] == 0
        
        
@pytest.mark.parametrize("num", [1,6,13,24,31,40])        
def test_bootstrap(num):
    data = rt.read_subject(num)
    data = np.array(data)

    for maniplulation_type in range(7):
        for soa in range(2):
            indices = equal.bootstrap_trials(data, maniplulation_type, soa, 10)
            
            assert (data[indices][:,1] == maniplulation_type).all()
            assert (data[indices][:,2] == soa).all()
    
@pytest.mark.parametrize("num", [1,6,13,24,31,40])           
def test_equalize_trials(num):
    data = rt.read_subject(num)
    data = util.subject_filter_medial(data)

    
    # define classifier function
    classifier = trial_classfier_creator(config="soa")
    
    # extract the indices of the two trials
    idx1, idx2 = subject_classifier(data, classifier)

    trial_counts1 = equal.trial_type_counter(data[idx1])
    trial_counts2 = equal.trial_type_counter(data[idx2])
    
    for i in range(len(trial_counts2)):
        diff = trial_counts1[i] - trial_counts2[i]
        # if there is no trials from this type in one of the classes, erase the in equivalent trials from the other indices list
        if trial_counts1[i] == 0:
            idx2 = equal.erase_trials(data, i, 0, idx2)
            continue
        if trial_counts2[i] == 0:
            idx1 = equal.erase_trials(data, i, 1, idx1)    
            continue
        
        
        idx1, idx2 = equal.equalize_trials(data, i, 10, idx1, idx2)
        
        trial_counts1 = equal.trial_type_counter(data[idx1])
        trial_counts2 = equal.trial_type_counter(data[idx2])
        assert trial_counts1[i] ==  trial_counts2[i]


@pytest.mark.parametrize("fun", [equal.soa_equalizer, equal.soa_equalizer_down])        
def test_trial_equal(fun):
    for i in range(cfg.num_of_participants):
        data = rt.read_subject(i+1)
        data = util.subject_filter_medial(data)

        # define classifier function
        classifier = trial_classfier_creator(config="soa")
        
        # extract the indices of the two trials
        first_class_idx, second_class_idx = subject_classifier(data, classifier)
        
        # equalize number of trials in each group if the soa condition was choose
        first_class_idx, second_class_idx = fun(data, first_class_idx, second_class_idx)
        
        count1 = equal.trial_type_counter(data[first_class_idx])
        count2 = equal.trial_type_counter(data[second_class_idx])
        
        assert count1 == count2
        
        assert (data[first_class_idx][:,2] == 1).all()
        
        assert (data[second_class_idx][:,2] == 0).all()
