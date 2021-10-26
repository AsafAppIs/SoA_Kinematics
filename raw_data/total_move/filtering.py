# -*- coding: utf-8 -*-
import numpy as np
import raw_data.configurations as cfg


# this function gets subject kinematic data and total movement data
# and returns subject kinematic data without trials with too big movements
# and the number of trials that where filtered
def filter_exaggerated_movements(subject_kinematics, movement):
    # throw away trials with exaggerated movement -> larger then 1
    # create a list of indices where the movement is too big
    exaggerated_lst = []
    
    for i, row in enumerate(movement):
        if sum(row[row > 1.5] > 1): # if there is point that moves more then 1 through one trial
            exaggerated_lst.append(i)
    
    # save the number of those numbers 
    num_exaggerated = len(exaggerated_lst)
    
    # delete them
    for i in sorted(exaggerated_lst, reverse=True):
        del subject_kinematics[i]
        
    return subject_kinematics, num_exaggerated



# this function gets subject kinematic data and total movement data
# and returns subject kinematic data without trials with too small movements or too big
# and the number of trials that where filtered
# the computation is based on the distal point in the index finger (8)
def filter_extreme_movements(subject_kinematics, movement):
    # create a list of indices where the movement is too big
    big_lst = []
    small_lst = []
    
    # compute threhold: 3 or 4 std from the mean
    mean_movement = np.mean(movement[:,8])
    std_movement = np.std(movement[:,8])
    min_threshold = mean_movement - std_movement * cfg.num_of_std_from_mean
    max_threshold = mean_movement + std_movement * cfg.num_of_std_from_mean
    
    # find all the indices where movement is smaller then threshold
    for i, row in enumerate(movement):
        if row[8] < min_threshold: # 3 std below mean
            small_lst.append(i)
        elif row[8] > max_threshold: # 3 std above mean
            big_lst.append(i)
            
    # save the number of those numbers
    num_of_big = len(big_lst)
    num_of_small = len(small_lst)
    
    # aggregate list in order to delete entries from the list
    extreme_lst = big_lst + small_lst
    
    # delete them
    for i in sorted(extreme_lst, reverse=True):
        del subject_kinematics[i]
        
    return subject_kinematics, num_of_big, num_of_small