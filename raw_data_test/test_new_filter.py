# -*- coding: utf-8 -*-
import pytest
import raw_data.read_raw as read
import raw_data.total_move.total_movement_extraction as tm_extract
import raw_data.movement_filtering.leap_filtering as l_fil

@pytest.fixture
def kinematic_data(request):
    return read.get_parsed_kinematic_data(1, request.param)


@pytest.mark.parametrize("kinematic_data, result_len, result_crazy, result_jump",[(98,96,2, 0),(101,96,2,2)], indirect=['kinematic_data'])
def test_new_filter(kinematic_data, result_len, result_crazy, result_jump):
    num_of_crazy, num_of_jumps = l_fil.leap_filtering(kinematic_data)
    
    
    
    assert num_of_crazy == result_crazy    
    assert num_of_jumps == result_jump
    assert len(kinematic_data) == result_len
    