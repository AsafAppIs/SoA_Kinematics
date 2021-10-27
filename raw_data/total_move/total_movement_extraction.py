import pandas as pd
import numpy as np
import raw_data.configurations as cfg
import raw_data.read_raw as read
from raw_data.total_move.filtering import filter_exaggerated_movements, filter_extreme_movements
from raw_data.total_move.movement_normalization import movement_hand_normalization 


# this function return the data  regards the total movement of all subjects
# the function has to modes: list mode and agrregated mode
def all_subjects_data(list_mode=True, normalize=False):
    subjects_lst = []
    for experiment, participant in cfg.all_participants:
        # extract kinematic data from file
        subject_kinematics = read.get_parsed_kinematic_data(experiment, participant)
        
        # caculate total movement
        movement, bad_trials = subject_total_movement(subject_kinematics, normalize=normalize)
        # add to list
        subjects_lst.append(movement)
        
    # if not on list_mode then aggregate and return
    if not list_mode:
        # concatenate all the movements from all participants together 
        whole_movement = np.concatenate(subjects_lst, axis=0)
        
        return whole_movement 
    
    # if list_mode then return list
    return subjects_lst



# this function extract and filters the data regards the total movement of subject
# it gets subject kinematic data and 
# returns numpy array that contain the total movement data from all trials after filtering
# array dimentions (num of trials, num of points)
def subject_total_movement(subject_kinematics, normalize=False):
    # define 2d array to conatin the total movement data from all trials
    # array dimentions (num of trials, num of points)
    movement = all_trials_total_movement(subject_kinematics, normalize=normalize)
    # filter anomalies
    movement, num_exaggerated = filter_exaggerated_movements(subject_kinematics, movement)
    
    
    # filter extremes
    #movement = all_trials_total_movement(subject_kinematics)
    movement, num_of_big, num_of_small = filter_extreme_movements(subject_kinematics, movement)
    
    # recreate movement array after filtering
    #movement = all_trials_total_movement(subject_kinematics)
    return movement, (num_exaggerated, num_of_big, num_of_small)



# this function extract the data regards the total movement of subject
# it gets subject kinematic data and 
# returns numpy array that contain the total movement data from all trials
# array dimentions (num of trials, num of points)
def all_trials_total_movement(subject_kinematics, normalize=False):
    # define 2d array to conatin the total movement data from all trials
    # array dimentions (num of trials, num of points)
    movement = np.zeros(shape=(len(subject_kinematics), 21))
    
    # iterate over trials and extract movement data
    for i, trial in enumerate(subject_kinematics):
        movement[i] = trial_total_movement(trial[1], normalize=normalize)

    return movement



# this function get a trial kinematic data frame
# and returns a vector that contain the total movement 
# done by all the 21 points recorded by leap
def trial_total_movement(trial_df, normalize=False):
    # take only the kinematic info from the file and convert it to numpy array
    kinematic_df = trial_df.iloc[:,9:]
    kinematic_array = np.array(kinematic_df, dtype=np.float)

    # if normalize flag is on: normalize the locations to hand base
    if normalize:
        kinematic_array = movement_hand_normalization(kinematic_array)
    

    # calculate the difference between each frame
    kinematic_diff = np.diff(kinematic_array, axis=0)

    # calculte the euclidian distance made
    # split to points, square, sum, sqrt, sum and combine
    kinematic_points = np.split(kinematic_diff, indices_or_sections=21, axis=1)
    for i in range(len(kinematic_points)):
        kinematic_points[i] = kinematic_points[i] ** 2
        kinematic_points[i] = np.sum(kinematic_points[i], axis=1)
        kinematic_points[i] = np.sqrt(kinematic_points[i])
        kinematic_points[i] = np.sum(kinematic_points[i])
        
    
    return np.array(kinematic_points)


