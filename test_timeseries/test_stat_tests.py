import numpy as np
import pytest
import timeseries_data.read_timeseries as rt
import timeseries_data.mean_ts.trial_classifier as trial_class
import timeseries_data.mean_ts.stat_anlaysis as stat_an
import timeseries_data.mean_ts.ts_statistics as ts_stat


classification_configurations = [(2, {0:1, 1:0}), (1, {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1}),
                                (1, {0:0, 1:1, 2:1, 3:1}), (1, {0:0, 4:1, 5:1, 6:1}), 
                                ([1,2], {(0,1):0, (1,1):1})]



def test_calculate_stats():
    data = np.zeros(shape=(10,2))
    data[0,1] = 1
    
    pval, mean_diff = stat_an.calculate_stats(data)
    
    assert pval == pytest.approx(0.3434,  0.001)
    assert mean_diff  == pytest.approx(-0.1)

@pytest.mark.parametrize("num", [1,4,7])
def test_subject_statisitical_analysis(num):
    data = rt.read_subject(num, test=True)
    ts_num = 1
    start=50
    end=80
    fun = ts_stat.part_trial_mean_creator(ts_num, start, end)

    results = ((-10,-3), (-3.5,-7), (-3.5,-5.5), (-3.5,-8.5), (-7,-8))
    for i, config in enumerate(classification_configurations):
        first_class, second_class = stat_an.subject_statisitical_analysis(data, fun, 
                                                                          idx=config[0], 
                                                                          class_dict=config[1])
        first_class = np.mean(first_class)
        second_class = np.mean(second_class)
        
        assert (first_class == results[i][0]).all()
        assert (second_class == results[i][1]).all() 
        
        

def test_all_mean_stat_comparison():
    ts_num = 1
    start=50
    end=80
    fun = ts_stat.part_trial_mean_creator(ts_num, start, end)
    results = ((-10,-3), (-3.5,-7), (-3.5,-5.5), (-3.5,-8.5), (-7,-8))
    for i, config in enumerate(classification_configurations):
        mean_results = stat_an.all_mean_stat_comparison(fun, idx=config[0], 
                                                        class_dict=config[1], range_limit=10, test=True)
        
        assert (mean_results[:,0] == results[i][0]).all()
        assert (mean_results[:,1] == results[i][1]).all() 


def test_subject_statistic_and_agency():
    subjects_data = stat_an.subject_statistic_and_agency(10, True)
    
    for subject in subjects_data:
        assert subject[0] == subject[1] == subject[2] == -7
        assert subject[3] == subject[4] == subject[5] == 3.5
        assert subject[6] == subject[7] == subject[8] == 2
        assert subject[9] == subject[10] == subject[11] == 5
        assert subject[30] == subject[31] == subject[32] == 1
        
def test_statistical_tables():
    subjects_data = stat_an.statistical_tables(10, True, 1)
    
    for subject in subjects_data:
        line = np.mean(subject, axis=1)
        assert line[0] == -10
        assert line[1] == -3
        assert line[2] == -10
        assert line[3] == -3
        assert line[4] == -10
        assert line[5] == -3
        assert line[6] == pytest.approx(47.142, 0.01)
        assert line[7] == pytest.approx(47.142, 0.01)
        assert line[8] == 30
        assert line[9] == 30
    

def test_statistical_tests():
    subjects_data = stat_an.statistical_tests(10, True, 1)            
    for subject in subjects_data:
        line = subject[3::2]
        assert line[0] == -7
        assert line[1] == -7
        assert line[2] == -7
        assert line[3] == 0
        assert line[4] == 0

