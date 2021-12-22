import pytest
import numpy as np
import timeseries_data.mean_ts.ts_statistics as ts_stat

def test_part_trial_mean_creator():
    data = np.zeros(shape=(1323))
    data[3+120+50:3+120+80] = 1
    data[3+240+50:3+240+80] = 1
    data[3+1200+40:3+1200+70] = 1
    
    
    ts_num = 1
    start=50
    end=80
    fun = ts_stat.part_trial_mean_creator(ts_num, start, end)
    
    assert fun(data) == 1
    
    ts_num = 2
    start=50
    end=80
    fun = ts_stat.part_trial_mean_creator(ts_num, start, end)
    
    assert fun(data) == 1
    
    ts_num = 10
    start=40
    end=70
    fun = ts_stat.part_trial_mean_creator(ts_num, start, end)
    
    assert fun(data) == 1
    
    
    
def test_derivative_peak_distance():
    data = np.zeros(shape=(1323))
    data[3+120+50:3+120+80] = -1
    data[3+240+50:3+240+85] = -1
    
    ts_num = 1
    fun = ts_stat.derivative_peak_distance(ts_num)
    
    assert fun(data) == 30
    
    ts_num = 2
    fun = ts_stat.derivative_peak_distance(ts_num)
    
    assert fun(data) == 35