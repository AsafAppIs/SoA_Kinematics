import numpy as np

import kinematic_data.configurations as cfg
from kinematic_data.kinematic_to_ts.derivative import calculate_derivative_timeseries
from kinematic_data.read_kinematic import read_subject


# this function gets a line that represent trial
# and return a matrix in which the rows represent location timeseries of different coordinates
def split_to_coordinates(trial):
    # get rid of the header
    trial_kinematics = trial[cfg.header_size:]
    
    # split to timeseries
    splited_trial_kinematics = np.array(np.split(trial_kinematics, indices_or_sections=cfg.num_of_original_ts))
    
    return splited_trial_kinematics


# this function gets a line that represent trial
# and return a timeseries representation of the trial
def trial_to_timeseries(trial):
    location_ts = split_to_coordinates(trial)
    
    # calculate velocity timesseries
    velocity_ts, total_velocity_ts = calculate_derivative_timeseries(location_ts)
    
    # calculate acceleration timesseries
    acceleration_ts, total_acceleration_ts  = calculate_derivative_timeseries(velocity_ts)
    
    # concat together
    line = np.concatenate([trial, velocity_ts, total_velocity_ts, acceleration_ts, total_acceleration_ts])
    
    return line


# this function gets kinematic dataframe and 
# returns a dataframe of timeseries representation
def subject_to_timeseries(kinematic_data):
    #import pdb; pdb.set_trace()
    # define list to contain new line representation
    ts_representation = []
    
    # cast the kinematic data into ndarray
    kinematic_data = np.array(kinematic_data)
    
    # iterate over lines
    for i in range(len(kinematic_data)):
        line = kinematic_data[i]
        new_line = trial_to_timeseries(line)
        ts_representation.append(new_line)
        
    # cast the timeseries data into ndarray
    timeseries_representation = np.array(ts_representation)
    
    return timeseries_representation


if __name__ == "__main__":
    for i in range(1, cfg.num_of_participants + 1):
        print(f"process participant {i}")
        kinematic_representation = read_subject(i)
        timeseries_representation = subject_to_timeseries(kinematic_representation)
        path = cfg.timeseries_path+"participant"+str(i)+".csv"
        np.savetxt(path, timeseries_representation, delimiter=',')
        


