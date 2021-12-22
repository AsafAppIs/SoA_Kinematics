import numpy as np
import pytest
import timeseries_data.read_timeseries as rt
import timeseries_data.configurations as cfg
import timeseries_data.mean_ts.subject_classifier as sub_c
from timeseries_data.mean_ts.trial_classifier import trial_classfier_creator



@pytest.mark.parametrize("participant_num, a_no, a_yes, m_no, m_yes, m_t, m_s, m_1, m_2, m_5, m_6",
                         [(1, 83, 171, 83, 171, 85, 86, 30, 28, 29, 28),
                          (33, 101, 131, 60, 172, 86, 86, 30, 27, 27, 30)])
def test_mean_classifier(participant_num, a_no, a_yes, m_no, m_yes, m_t, m_s, m_1, m_2, m_5, m_6):
    data = rt.read_subject(participant_num)
    data = np.array(data)
    
    # soa test
    classifier = trial_classfier_creator(config="soa")
    first_idx, second_idx = sub_c.subject_classifier(data, classifier)
    assert len(first_idx) == a_no
    assert len(second_idx) == a_yes
    
    # manipulation test
    classifier = trial_classfier_creator(config="manipulation")
    first_idx, second_idx = sub_c.subject_classifier(data, classifier)
    assert len(first_idx) == m_no
    assert len(second_idx) == m_yes
    
    # temporal manipulation test
    classifier = trial_classfier_creator(config="manipulation_t")
    first_idx, second_idx = sub_c.subject_classifier(data, classifier)
    assert len(first_idx) == m_no
    assert len(second_idx) == m_t
    
    # spatial manipulation test
    classifier = trial_classfier_creator(config="manipulation_s")
    first_idx, second_idx = sub_c.subject_classifier(data, classifier)
    assert len(first_idx) == m_no
    assert len(second_idx) == m_s
    
    # 1-2 manipulation test
    class_dict = {1:0, 2:1}
    classifier = trial_classfier_creator(idx=1, class_dict=class_dict)
    first_idx, second_idx = sub_c.subject_classifier(data, classifier)
    assert len(first_idx) == m_1
    assert len(second_idx) == m_2
    
    # 5-6 manipulation test
    class_dict = {5:0, 6:1}
    classifier = trial_classfier_creator(idx=1, class_dict=class_dict)
    first_idx, second_idx = sub_c.subject_classifier(data, classifier)
    assert len(first_idx) == m_5
    assert len(second_idx) == m_6
    
    