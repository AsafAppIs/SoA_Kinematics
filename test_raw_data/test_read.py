# -*- coding: utf-8 -*-
import pytest
import raw_data.read_raw as read

@pytest.fixture
def kinematic_data(request):
    return read.get_parsed_kinematic_data(1, request.param)


@pytest.mark.parametrize("kinematic_data, result",[(98,98),(99,97)], indirect=['kinematic_data'])
def test_length(kinematic_data, result):
    assert len(kinematic_data) == result
    

