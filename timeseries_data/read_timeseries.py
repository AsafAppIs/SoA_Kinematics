import pandas as pd
import timeseries_data.configurations as cfg






# this function gets participant number and returns dataframe of kinematics data
def read_subject(participant):
    # define the path to the appropriate participant
    path = cfg.timeseries_path + "participant" + str(participant) + ".csv"
    
    # raed the data
    data = pd.read_csv(path, header=None)
        
    return data 