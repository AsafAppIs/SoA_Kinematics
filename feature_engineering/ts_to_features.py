import numpy as np 
import pandas as pd
import tsfresh as ts
import feature_engineering.configurations as cfg
import timeseries_data.util.util as util
import timeseries_data.read_timeseries as r_ts
from tsfresh.utilities.dataframe_functions import impute


def to_ts_features(data):
    extracted_features = ts.extract_features(data, column_id="id")
    impute(extracted_features)
    return extracted_features 


# tranfrom each trial to suitable df for tsfresh
def ts_format(trial_data, idx):
    # tranfrom data into columns represenatation
    trial_data = trial_data.T
    
    # to DataFrame
    trial_df = pd.DataFrame(trial_data)
    
    # assign id value
    trial_df['id'] = idx
    
    return trial_df

# this function transfrom the timeseries data into dataframe in a format that suit the api of tsfresh
def construct_data_to_tsformat(data):
    # split each trial to different ts
    data = [util.split(trial) for trial in data]
    
    # build DataFrame in format that suits tsfresh
    data = [ts_format(trial, i) for i, trial in enumerate(data)]
    
    # concat all the dataframes together
    data = pd.concat(data, axis=0)
    
    return data

# transfrom the timeseries representation into feature representation using tsfresh package 
# after, the function will save the feature respresentation
def participant_to_features(participant, test=False, only_distal=True):
    # read timeseries data 
    data = r_ts.read_subject(participant, test=test)
    
    # take only distal timeseries
    data = util.subject_filter_medial(data)
    
    ts_dataframe = construct_data_to_tsformat(data)
    
    features_rep = to_ts_features(ts_dataframe)
    
    header = pd.DataFrame(data[:, :cfg.header_size])
    
    features_rep = pd.concat((header, features_rep), axis=1)
    
    features_rep.to_csv(cfg.full_feature_path+"participant"+str(participant)+".csv", index=False)
    
if __name__ == "__main__":
    for i in range(5,42):
        participant_to_features(i)

