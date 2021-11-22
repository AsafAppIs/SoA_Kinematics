import numpy as np
from timeseries_data.util.util import get_indices
from random import choice, seed


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
            idx2 = erase_trials(subject_data, i, 1, idx2)
        elif trial_counts2[i] == 0:
            idx1 = erase_trials(subject_data, i, 0, idx1)    
        
        # if there is a differnce, the function will bootsprap more trials for the minority class
        elif diff == 0:
            continue
        elif diff > 0:
            idx2 += bootstrap_trials(subject_data, i, 1, diff)
        elif diff < 0:
            idx1 += bootstrap_trials(subject_data, i, 0, -diff)
            
    return idx1, idx2
        
        
        
        