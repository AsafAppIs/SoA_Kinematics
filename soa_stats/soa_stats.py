import pandas as pd
import numpy as np
from scipy import stats

import timeseries_data.configurations as cfg
from timeseries_data.read_timeseries import read_subject
from timeseries_data.util.util import trial_filter_creator

# this function gets subject header data and return sensitivity and bias measures
# after stanislaw & Todorov (1999) correction
def signal_detection(correct, manipulation):
    # manipulation array is 0 when manipulated and when when not
    to_hit = correct.astype(np.int8) + manipulation.astype(np.int8)
    
    # hit is trial where the subject was correct and there was no manipulation, 
    # meaning the to_hits array will be equal 2
    hits = np.sum(to_hit == 2)
    
    
    agency_trials = np.sum(manipulation)
    hit = (hits + .5)/(agency_trials + 1)
    
    # all trials stats
    # miss is trial where the subject was wrong and there was manipulation, meaning the to_hit array would be zero
    miss = np.sum(to_hit == 0)
    no_agency_trials = np.sum(manipulation == 0)
    #false_alarm = miss/no_response
    false_alarm = (miss + .5)/(no_agency_trials + 1)
    
    # all trials sdt
    hit_z = stats.norm.ppf(hit)
    fa_z = stats.norm.ppf(false_alarm)
    dprime = hit_z - fa_z
    criterion = -(hit_z + fa_z)/2
    
    
    
    return dprime, criterion


# this functions gets subject data and calculate the sensitivity (dprime) and the bias
def signal_detection_calculations(subject):
    manipulated = subject[subject[:,1] > 0]
    not_manipulated = subject[subject[:,1] == 0]
    
    fa = manipulated[manipulated[:,2] == 1]
    hits = not_manipulated[not_manipulated[:,2] == 1]
    
    hit_rate = (len(hits) + 0.5) / (len(not_manipulated) + 1)   
    fa_rate = (len(fa) + 0.5) / (len(manipulated) + 1)

    hit_z = stats.norm.ppf(hit_rate)
    fa_z = stats.norm.ppf(fa_rate)
    dprime = hit_z - fa_z
    criterion = -(hit_z + fa_z)/2
    
    return dprime, criterion
    
    
#this function gets subject data and return the general stats:
# agency rate, accuracy and sensitivity 
def subject_agency_statistics(subject_data):
    # take only the header part of the data
    header = subject_data[:, :cfg.header_size]
    
    #calculate the agency rate
    agency_rate = np.sum(header[:,2])
    agency_rate /= len(subject_data)
    
    # calculate accuracy
    manipulation = header[:,1] == 0
    correct = header[:,2] == manipulation
    accuracy_rate = np.sum(correct) / len(correct)
    
    dprime, criterion = signal_detection(correct, manipulation)
    
    return agency_rate, accuracy_rate, dprime, criterion

    
def subject_sensitivity_stats(subject_num):
    results = []
    data = np.array(read_subject(subject_num))
    for config in cfg.sensitivity_configurations:
        # create filtering function
        filter_fun = trial_filter_creator(config[1], config[2])
        # filter the data
        relevant_data = np.array([line for line in data if filter_fun(line)])
        # calculate dprime & criterion
        dprime, criterion = signal_detection_calculations(relevant_data)
        
        # add results to results list
        results.append(dprime)
        results.append(criterion)
    
    return results

def all_sensitivity_stats():
    results = []
    for i in range(cfg.num_of_participants):
        results.append(subject_sensitivity_stats(i+1))
        
    results = pd.DataFrame(results, columns=cfg.sensitivity_names_cols)
    
    results.to_csv(cfg.sensitivity_path + "sensitivity.csv", index=None)
    
    return results

#x = all_sensitivity_stats()
