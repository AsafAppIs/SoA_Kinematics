import numpy as np
from copy import deepcopy as deepc
from timeseries_data.util.util import get_indices
from random import choice, seed
import timeseries_data.configurations as cfg
from timeseries_data.read_timeseries import read_subject
def counts_format(unique, counts):
    for i in range(7):
        if i not in unique:
            counts.insert(i,0)
    return counts

# this function gets many trials data and return counts of trial types
def trial_type_counter(trials_data):
    types = trials_data[:,1]
    unique, counts = np.unique(types, return_counts=True)
    
    counts = list(counts)
    
    counts = counts_format(unique, counts)
    
    return counts




# this function get trial data, specific trial, soa response and list of indices. 
# the function finds the indeices of trials that meet the conditions and erase them from the list
def erase_trials(subject_data, trial_type, is_soa, indices):
    # clone indices list in order to protect the original indices list
    indices = deepc(indices)
    # get indices to erase
    erase_idx = get_indices(subject_data, trial_type, is_soa)
    
    # erase them
    for idx in erase_idx:
        indices.remove(idx)
    
    return indices


# this function gets  trial data, specific trial, soa response and a number of new required trials
# the function return corresponding number of indices of trials that meet the conditions
def bootstrap_trials(subject_data, trial_type, is_soa, number_of_new_trials):
    # reproduction
    seed(10)
    
    indices = get_indices(subject_data, trial_type, is_soa)
    
    if indices.size == 0:
        return []
    
    new_indices = [choice(indices) for i in range(number_of_new_trials)]
    
    return new_indices 



# this function gets subject data array and two list of indices thats represnt trials with and without SoA
# in order to deconfound this split, this function equalize the number of trials from the same domainXlevel in both lists
# by bootstraping more trials from the minority list. if in one ilst there is no trials at all from a specific level
# the function will erase all the instances of this class from the other list
def soa_equalizer(subject_data, idx1, idx2):
    # calculate the trial counts for each class
    trial_counts1 = trial_type_counter(subject_data[idx1])
    trial_counts2 = trial_type_counter(subject_data[idx2])
    
    for i in range(len(trial_counts2)):
        diff = trial_counts1[i] - trial_counts2[i]
        # if there is no trials from this type in one of the classes, erase the in equivalent trials from the other indices list
        if trial_counts1[i] == 0:
            idx2 = erase_trials(subject_data, i, 0, idx2)
        elif trial_counts2[i] == 0:
            idx1 = erase_trials(subject_data, i, 1, idx1)    
          
        # if there is a differnce, the function will bootsprap more trials for the minority class
        elif diff == 0:
            continue
        elif diff > 0:
            idx2 += bootstrap_trials(subject_data, i, 0, diff)
        elif diff < 0:
            idx1 += bootstrap_trials(subject_data, i, 1, -diff)
           
    return idx1, idx2
     


# this function gets subject, indices to draw from, number of samples to draw each time and number of draws
# the function bootstrap samples and return the average.
def mean_samples(subject_data, source_idx, size_of_draw, num_of_bootstraps):
    data = []
    seed(100)
    for i in range(num_of_bootstraps):
        new_indices = [choice(source_idx) for i in range(size_of_draw)]
        new_data = subject_data[new_indices]
        data.append(new_data)
        
    data = np.array(data)
    data_mean = np.mean(data, axis=0)
    data_mean = data_mean[:, cfg.header_size:]
    return data_mean

# this function gets subject data, number that represent trials type and number of bootstraps for averaging
# the function will find out if there is inequelity between classes, and if so the function will
# draw n "minority" samples from majority, average them, edit the data and update the idx list
def equalize_trials(subject_data, trial_type, num_of_bootstraps, idx1, idx2):
    idx_noagency = [idx for idx in range(len(subject_data)) if subject_data[idx,1]==trial_type and subject_data[idx,2]==0]
    idx_agency = [idx for idx in range(len(subject_data)) if subject_data[idx,1]==trial_type and subject_data[idx,2]==1]
    
    diff = len(idx_agency) - len(idx_noagency)
    
    if diff > 0:
        # extract "new" samples from existing samples mean
        mean_new_samples = mean_samples(subject_data, idx_agency, len(idx_noagency), num_of_bootstraps)
        
        # update data array with the new samples
        for i in range(len(mean_new_samples)):
            subject_data[idx_agency[i]][cfg.header_size:] = mean_new_samples[i]
            
        # delete the rest of the indices from idx array
        for val in idx_agency[len(mean_new_samples):]:
            idx1.remove(val)
            
    elif diff < 0:
        # extract "new" samples from existing samples mean
        mean_new_samples = mean_samples(subject_data, idx_noagency, len(idx_agency), num_of_bootstraps)
        # update data array with the new samples
        for i in range(len(mean_new_samples)):
            subject_data[idx_noagency[i]][cfg.header_size:] = mean_new_samples[i]
            
        # delete the rest of the indices from idx array
        for val in idx_noagency[len(mean_new_samples):]:
            idx2.remove(val)
            
    return idx1, idx2

# this function gets subject data array and two list of indices thats represnt trials with and without SoA
# in order to deconfound this split, this function equalize the number of trials from the same domainXlevel in both lists
# by downsampling trials from the majority class (sampling 15 points from majority and avreging). if in one ilst there is no trials at all from a specific level
# the function will erase all the instances of this class from the other list
def soa_equalizer_down(subject_data, idx1, idx2, num_of_bootstraps=10):
    # calculate the trial counts for each class
    trial_counts1 = trial_type_counter(subject_data[idx1])
    trial_counts2 = trial_type_counter(subject_data[idx2])
    
    for i in range(len(trial_counts2)):
        diff = trial_counts1[i] - trial_counts2[i]
        # if there is no trials from this type in one of the classes, erase the in equivalent trials from the other indices list
        if trial_counts1[i] == 0:
            idx2 = erase_trials(subject_data, i, 0, idx2)
            continue
        if trial_counts2[i] == 0:
            idx1 = erase_trials(subject_data, i, 1, idx1)    
            continue
        
        
        idx1, idx2 = equalize_trials(subject_data, i, num_of_bootstraps, idx1, idx2)
            
    return idx1, idx2




        
        