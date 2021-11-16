import numpy as np
import timeseries_data.configurations as cfg
from timeseries_data.read_timeseries import read_subject
from timeseries_data.util import split

# that function gets ts representation of trial, 
# calculate and return the correlation matrix between the differnt timeseries
def trial_correlation(trial):
    # split each ts to different row
    row_ts = split(trial)
    
    # calculate correlation
    corr = np.corrcoef(row_ts)
    
    return corr


# that functions get subject data in timeseries representation
# calculate and return the correlation matrix between the differnt timeseries
def subject_correlation(subject_ts):
    # define list that will contain the correlation matrices
    corr_lst = []
    
    # run over the trials
    for i in range(len(subject_ts)):
        # calculate trial correlation matrix
        corr = trial_correlation(subject_ts[i])
        
        # add dimention to corr in order to concatenate in the end
        corr = np.expand_dims(corr, axis=0)
        
        # add the correlation matrix into the container
        corr_lst.append(corr)
        
    total_corr = np.concatenate(corr_lst, axis=0)
    
    # return the mean correlation
    return np.mean(total_corr, axis=0)
    

# this function calculate and return he correlation matrix between the differnt timeseries in all the data
def all_correlation():
    # define list that will contain the correlation matrices
    corr_lst = []
    
    # run over subjects
    for i in range(1, cfg.num_of_participants + 1):
        # read subject data
        subject_ts = read_subject(i)
        
        subject_ts = np.array(subject_ts)
        
        # calculate subject correlation matrix
        corr = subject_correlation(subject_ts)
        
        # add dimention to corr in order to concatenate in the end
        corr = np.expand_dims(corr, axis=0)
        
        # add the correlation matrix into the container
        corr_lst.append(corr)
        
    
    total_corr = np.concatenate(corr_lst, axis=0)
    
    # return the mean correlation
    return np.mean(total_corr, axis=0)