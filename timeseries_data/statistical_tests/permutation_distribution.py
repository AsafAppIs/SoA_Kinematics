import numpy as np

import timeseries_data.configurations as cfg
import timeseries_data.util.comparisons as cmp
from timeseries_data.read_timeseries import read_subject
from timeseries_data.mean_ts.subject_classifier import conditional_mean_ts, all_mean_ts
import timeseries_data.util.util as util


def create_distribution(participant_num, distance_function, num_of_permutations):
    # read participant data
    data = read_subject(participant_num)
    data = util.subject_filter_medial(data)
    
    # create distibutation container
    disributions = np.zeros((cfg.only_distal, num_of_permutations))
    for i in range(num_of_permutations):        
        first_class_mean, _, second_class_mean, _= conditional_mean_ts(data, config="random")
        for ts in range(cfg.only_distal):
            first = util.get_timeseries(first_class_mean, ts)
            second = util.get_timeseries(second_class_mean, ts)
            disributions[ts,i] = distance_function(first, second)
            
    return disributions