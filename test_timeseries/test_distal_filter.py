import numpy as np
import pytest
import timeseries_data.read_timeseries as rt
import timeseries_data.util.util as ut
import timeseries_data.configurations as cfg

@pytest.mark.parametrize("num", [1,6,13,24,31,40])
def test_distal_filter(num):
    data = rt.read_subject(num)
    d_data = ut.subject_filter_medial(data)
    
    data= np.array(data)
    
    assert (data[:, :cfg.header_size] == d_data[:, :cfg.header_size]).all()
    
    for original, new in [(3,0), (5,2), (21,10)]:
        original_start = cfg.header_size + cfg.ts_length * original
        original_end = cfg.header_size + cfg.ts_length * (original+1)
        
        new_start = cfg.header_size + cfg.ts_length * new
        new_end = cfg.header_size + cfg.ts_length * (new+1)
 
        assert (data[:, original_start: original_end] == d_data[:, new_start:new_end]).all()