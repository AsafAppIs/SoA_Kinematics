import numpy as np
import timeseries_data.configurations as cfg
from timeseries_data.mean_ts.subject_classifier import conditional_mean_ts
import timeseries_data.util.util as util
# this function gets subject data and a comparison function
# and return matrix of comparison between all timeseries with respect to different classifications 
def timeseries_compare(subject_data, comparison_fun):
    # define a matrix in shape of num_of_comparisonsXnum_of_ts
    num_of_ts = int((subject_data.shape[1] - 3) / cfg.ts_length)
    num_of_comparisons = len(cfg.class_configurations)
    match_matrix = np.zeros((num_of_comparisons, num_of_ts))
    
    # iterate over the configuration
    for i, conf in enumerate(cfg.class_configurations):
        # calculate the mean timeseries for the codition
        if isinstance(conf, tuple):
            first_class_mean, first_class_ste, second_class_mean, second_class_ste = conditional_mean_ts(subject_data, idx=conf[1], class_dict=conf[2])
        else:
            first_class_mean, first_class_ste, second_class_mean, second_class_ste = conditional_mean_ts(subject_data, config=conf)
            
        
        # iterate over the timeseries and compare between them
        for j in range(num_of_ts):
            first_ts = util.get_timeseries(first_class_mean, j)
            second_ts = util.get_timeseries(second_class_mean, j)
            match = comparison_fun(first_ts, second_ts)
            match_matrix[i][j] = match
            
    return match_matrix
