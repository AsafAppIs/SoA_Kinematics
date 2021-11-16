import pytest
import kinematic_data.read_kinematic as rk
import timeseries_data.read_timeseries as rt
import timeseries_data.configurations as cfg

def test_kinematic_ts_congruency():
    for i in range(cfg.num_of_participants):
        kinematic_data = rk.read_subject(i+1)
        ts_data = rt.read_subject(i+1)
        
        kinematic_data = kinematic_data.iloc[:, :3]
        ts_data = ts_data.iloc[:, :3]
        
        assert kinematic_data.equals(ts_data)
