import numpy as np

import timeseries_data.util.util as util
 

def part_trial_mean_creator(ts_num, start, end):
    def fun(trial):
        ts = util.get_timeseries(trial, ts_num)
        part_ts = ts[start:end]
        mean_part_ts = np.mean(part_ts)
        return mean_part_ts 
    
    return fun