import numpy as np
import timeseries_data.configurations as cfg

# that function gets ts representation of trial
# split the different ts, and return a matrix in whice each line represent different ts
def split(trial_ts):
    # take only the kinematic data
    kinematics = trial_ts[cfg.header_size:]
    
    # split to rows
    row_rep = kinematics.reshape((cfg.num_of_ts, cfg.ts_length))
    
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
def get_timeseries(line, idx):
    # calculate offset
    offset = len(line) % cfg.ts_length
    start = offset + idx*cfg.ts_length
    end = start + cfg.ts_length
    return line[start:end]
    
