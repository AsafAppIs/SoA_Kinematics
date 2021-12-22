import numpy as np
import pytest
import timeseries_data.read_timeseries as rt
import timeseries_data.util.util as ut
import timeseries_data.configurations as cfg




@pytest.mark.parametrize("num", [1,6,13,24,31,40])
def test_split(num):
    data = rt.read_subject(num)
    data = np.array(data)
    
    line = data[0]
    
    split_rep = ut.split(line)
    
    # check the shape of the output
    assert split_rep.shape == (cfg.num_of_ts, cfg.ts_length)
    
    # check the content
    for i in range(cfg.num_of_ts):
        ts = line[cfg.header_size + i * cfg.ts_length: cfg.header_size + (i+1) * cfg.ts_length]
        assert (ts == split_rep[i]).all()


@pytest.mark.parametrize("num", [1,6,13,24,31,40])
def test_distal_filter(num):
    data = rt.read_subject(num)
    d_data = ut.subject_filter_medial(data)
    
    data = np.array(data)
    
    # check that the header stay the same
    assert (data[:, :cfg.header_size] == d_data[:, :cfg.header_size]).all()
    
    # check the new shape of the data
    assert d_data.shape == (len(data), cfg.header_size + cfg.only_distal*cfg.ts_length)
    
    for original, new in [(3,0), (5,2), (21,10)]:
        original_start = cfg.header_size + cfg.ts_length * original
        original_end = cfg.header_size + cfg.ts_length * (original+1)
        
        new_start = cfg.header_size + cfg.ts_length * new
        new_end = cfg.header_size + cfg.ts_length * (new+1)
 
        assert (data[:, original_start: original_end] == d_data[:, new_start:new_end]).all()
 
        
 
@pytest.mark.parametrize("num", [1,6,13,24,31,40])      
def test_get_ts(num):
    data = rt.read_subject(num)
    data = np.array(data)
    
    line = data[0]
    
    for i in range(cfg.num_of_ts):
        ts = line[cfg.header_size + i * cfg.ts_length: cfg.header_size + (i+1) * cfg.ts_length]
        get_ts = ut.get_timeseries(line, i)
        
        assert (ts == get_ts).all()
    
    

    
@pytest.mark.parametrize("num", [1,6,13,24,31,40])      
def test_trial_filter_creator(num):
    data = rt.read_subject(num)
    data = np.array(data)

    
    
    for filter_cfg in cfg.sensitivity_configurations:
        idx = filter_cfg[1]
        dic = filter_cfg[2]
        fun = ut.trial_filter_creator(idx, dic)
        
        dic_keys = list(dic.keys())
        
        for line in data:
            key = fun(line[:cfg.header_size])
            
            current_header = line[idx]
            if not isinstance(current_header , np.float64):
                current_header  = tuple(current_header )
            assert (current_header  in dic_keys) == key