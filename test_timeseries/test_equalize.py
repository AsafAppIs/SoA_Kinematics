import pytest
import timeseries_data.read_timeseries as rt
import timeseries_data.configurations as cfg
import timeseries_data.util.util as util
import timeseries_data.util.soa_equal as equal
from timeseries_data.mean_ts.trial_classifier import trial_classfier_creator
from timeseries_data.mean_ts.subject_classifier import subject_classifier


def test_trial_equal():
    for i in range(cfg.num_of_participants):
        print(i)
        data = rt.read_subject(i+1)
        data = util.subject_filter_medial(data)

        # define classifier function
        classifier = trial_classfier_creator(config="soa")
        
        # extract the indices of the two trials
        first_class_idx, second_class_idx = subject_classifier(data, classifier)
        
        # equalize number of trials in each group if the soa condition was choose
        first_class_idx, second_class_idx = equal.soa_equalizer(data, first_class_idx, second_class_idx)
        
        count1 = equal.trial_type_counter(data[first_class_idx])
        count2 = equal.trial_type_counter(data[second_class_idx])
        
        assert count1 == count2
        
        assert (data[first_class_idx][:,2] == 0).all()
        
        assert (data[second_class_idx][:,2] == 1).all()
