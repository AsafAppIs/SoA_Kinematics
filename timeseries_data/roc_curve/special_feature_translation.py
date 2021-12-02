import pandas as pd
import numpy as np

import timeseries_data.mean_ts.ts_statistics as ts_stat
import timeseries_data.configurations as cfg
from timeseries_data.read_timeseries import read_subject
import timeseries_data.util.util as util


# this function gets subject data and transform it into special featrues (y_loc, z_loc, total_acc) representation
def to_special_features(subject_data):
    # calculate the first feature
    ts_num = 1
    start=50
    end=80
    fun = ts_stat.part_trial_mean_creator(ts_num, start, end)
    
    first_feature = np.array([fun(trial) for trial in subject_data])
    first_feature  = np.reshape(first_feature, (-1, 1))
    
    
    # calculate the second feature
    ts_num = 2
    start=50
    end=80
    fun = ts_stat.part_trial_mean_creator(ts_num, start, end)
    
    second_feature = np.array([fun(trial) for trial in subject_data])
    second_feature   = np.reshape(second_feature , (-1, 1))
    
    
    # calculate the first feature
    ts_num = 10
    start=40
    end=70
    fun = ts_stat.part_trial_mean_creator(ts_num, start, end)
    
    third_feature = np.array([fun(trial) for trial in subject_data])
    third_feature = np.reshape(third_feature, (-1, 1))


    new_representation = np.concatenate([subject_data[:, :cfg.header_size], first_feature, second_feature
                                         , third_feature], axis=1)    
    
    return new_representation 


if __name__ == "__main__":
    for i in range(cfg.num_of_participants):
        print(i)
        # read data
        data = read_subject(i+1)
        data = util.subject_filter_medial(data) 
        
        data = to_special_features(data)
        
        df = pd.DataFrame(data)
        
        df.to_csv(cfg.special_feature_path + "participant" + str(i+1) + ".csv", header=None, index=None)
        