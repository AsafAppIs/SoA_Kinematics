import numpy as np
import timeseries_data.configurations as cfg
from scipy.fft import fft


# that function gets ts representation of trial
# split the different ts, and return a matrix in whice each line represent different ts
def split(trial_ts, is_freq=False):
    # define timeseries length
    length = int(cfg.ts_length/2) if is_freq else cfg.ts_length
    
    # take only the kinematic data
    kinematics = trial_ts[cfg.header_size:]
    
    # split to rows
    row_rep = kinematics.reshape((-1, length))
    
    return row_rep



# this function filters out the medial timeseries (0,1,2,6,7,8,14,15,16) from trial
def trial_filter_medial(trial_data):
    # split trial to rows
    row_rep = split(trial_data)
    
    # chose only distal sensor timeseries
    only_distal = row_rep[cfg.distal_idx]
    
    # concatenate the distal timeseries with the header
    only_distal_line = np.concatenate(only_distal , axis=0)
    only_distal_line = np.concatenate((trial_data[:cfg.header_size], only_distal_line) , axis=0)
    
    return only_distal_line 
    
    


# this function filters out the medial timeseries (0,1,2,6,7,8,14,15,16) from subject data
def subject_filter_medial(subject_data):
    # transform the data to ndarray
    subject_data = np.array(subject_data)   
    
    # create a list to contain the new lines
    distal_data_lst = []
    
    # iterate over the lines and transform them
    for line in subject_data:
       distal_data_lst.append(trial_filter_medial(line))
       
    # transfomr the list to ndarray
    
    distal_data = np.array(distal_data_lst)
    
    return distal_data 
    


# this function gets trial and timeseries index 
# and return only the timeseries in the corresponding index
def get_timeseries(line, idx, is_freq=False):
    # define timeseries length
    length = int(cfg.ts_length/2) if is_freq else cfg.ts_length
    
    # calculate offset
    offset = len(line) % length
    start = offset + idx*length
    end = start + length
    return line[start:end]


# this function gets a signal and transform it into the frequency domain
def to_frequency(signal):
    yf = fft(signal)
    yf[0] = 0
    yf = abs(yf[:int(cfg.ts_length/2)])
    
    return yf


# this function get a trial and transform each ts into frequency domain
def trial_to_frequency(trial):
    # split data to trials
    ts_rows = split(trial)
    
    # transfrom to frequency
    ts_freq = [to_frequency(row) for row in ts_rows]
    
    #concatenate
    new_trial = np.concatenate(ts_freq)
    new_trial = np.concatenate((trial[:cfg.header_size], new_trial))
    
    return new_trial 


#  this function get a subject and transform each ts into frequency domain
def subject_to_frequency(subject_data):
    subject_data_freq = np.array(list(map(trial_to_frequency, subject_data)))
    return subject_data_freq 
    
    

# this function gets subject data, trial type and soa indicators
# and return indices of trials that match the trial type and the soa answer 
def get_indices(data, trial_type, soa):
    idx = np.where((data[:,1] == trial_type) & (data[:,2] == soa))
    return idx[0]




# this function create a 'trial filter' function that filter trials from subject data
# this function gets index and dictionary to create custom filtering function
# the function return a filter thats return True for desireable trials anf False for undesireable trials 
def trial_filter_creator(idx=0, class_dict=False):    
    # define the classifier function
    def trial_filter(trial_header):
        if isinstance(idx, int):
            key = trial_header[idx]
        else:
            key = tuple(trial_header[idx])
        return class_dict.get(key, False)
    
    return trial_filter