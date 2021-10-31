# -*- coding: utf-8 -*-
import pytest
import raw_data.read_raw as read
import raw_data.total_move.total_movement_extraction as tm_extract
import raw_data.total_move.total_movement_stats as tm_stats
import raw_data.total_move.filtering as ftr

@pytest.fixture
def kinematic_data(request):
    return read.get_parsed_kinematic_data(1, request.param)


@pytest.mark.parametrize("kinematic_data, result_len, result_trials",[(98,92,2),(99,88,3)], indirect=['kinematic_data'])
def test_length_after_filtering(kinematic_data, result_len, result_trials):
    movement, num_of_bad_trials = tm_extract.subject_total_movement(kinematic_data)
    
    assert len(kinematic_data) == result_len
    
    assert num_of_bad_trials[0] == result_trials
    
    assert num_of_bad_trials[2] == 0
    
    assert num_of_bad_trials[1] == result_trials * 2
    

@pytest.mark.parametrize("list_mode, result_con, result_obj",[(True,2,92),(False,180,21)])
def test_all_participant(list_mode, result_con, result_obj):
    data = tm_extract.all_subjects_data(list_mode)
    
    assert len(data) == result_con
    
    assert len(data[0]) == result_obj

    
@pytest.mark.parametrize("kinematic_data",[98,99], indirect=['kinematic_data'])   
def test_total_movement_stats_calculation(kinematic_data):
    movement, num_of_bad_trials = tm_extract.subject_total_movement(kinematic_data)
    movement_stats = tm_stats.total_movement_stats_calculation(movement)
    
    assert sum(movement_stats[1]) == pytest.approx(0)
    

    assert movement_stats[0,7] * 2 == pytest.approx(movement_stats[0,8])
    

@pytest.mark.parametrize("kinematic_data, ratio",[(98, 1),(100, .9)], indirect=['kinematic_data'])   
def test_normalizing(kinematic_data, ratio):
    clean_movement, num_of_bad_trials = tm_extract.subject_total_movement(kinematic_data, True)
    movement, num_of_bad_trials = tm_extract.subject_total_movement(kinematic_data)
    
    
    movement_stats = tm_stats.total_movement_stats_calculation(movement)
    clean_movement_stats = tm_stats.total_movement_stats_calculation(clean_movement)
    
    assert clean_movement[0,0] == 0
    assert movement_stats[0,7] * ratio == pytest.approx(clean_movement_stats[0,7])
    
    
