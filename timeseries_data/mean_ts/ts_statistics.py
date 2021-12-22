import numpy as np

import timeseries_data.util.util as util
 

def part_trial_mean_creator(ts_num, start, end):
    def fun(trial):
        ts = util.get_timeseries(trial, ts_num)
        part_ts = ts[start:end]
        mean_part_ts = np.mean(part_ts)
        return mean_part_ts 
    
    return fun


def derivative_peak_distance(ts_num):
    def fun(trial):
        ts = util.get_timeseries(trial, ts_num)
        diff_ts = np.diff(ts)
        min_idx = np.where(diff_ts == np.min(diff_ts))[0][0]
        max_idx = np.where(diff_ts == np.max(diff_ts))[0][0]
        distance = max_idx - min_idx
        return distance 
    
    return fun