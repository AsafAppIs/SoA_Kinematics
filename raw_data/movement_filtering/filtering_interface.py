# -*- coding: utf-8 -*-
import numpy as np
import raw_data.configurations as cfg
import raw_data.read_raw as read
from raw_data.movement_filtering.leap_filtering import leap_filtering
from raw_data.movement_filtering.participant_movement_filtering import movement_filtering
from raw_data.normalization.hand_base_normalize import hand_base_normaliztion


def all_subjects_filter(list_mode=True, normalize=False):
    subjects_lst = []
    total_bad_trials = np.zeros((1,3))
    for experiment, participant in cfg.all_participants:
        if participant == 4:
            x=2
        # extract kinematic data from file
        subject_kinematics = read.get_parsed_kinematic_data(experiment, participant)
        
        # normalize
        if normalize:
            hand_base_normaliztion(subject_kinematics)
            
        # filter
        #total_bad_trials += filter_subject(subject_kinematics)
        print(f"Experiment {experiment} participant {participant:2}: {filter_subject(subject_kinematics)}")
        
        # add to list
        subjects_lst.append(subject_kinematics)
        
    # if not on list_mode then aggregate and return
    if not list_mode:
        # concatenate all the movements from all participants together 
        whole_movement = np.concatenate(subjects_lst, axis=0)
        
        return whole_movement 
    
    #print(total_bad_trials)
    # if list_mode then return list
    return subjects_lst
 


def filter_subject(kinematics):
    num_of_crazy, num_of_jumps = leap_filtering(kinematics)
    num_of_unproper = movement_filtering(kinematics)
    return np.array((num_of_crazy, num_of_jumps, num_of_unproper))
    
if __name__ == "__main__":    
    experiment_num = 2
    participant_num = 4
    trial_num = 24
    
    kinematic = read.get_parsed_kinematic_data(experiment_num, participant_num)
    hand_base_normaliztion(kinematic)
    #print(filter_subject(kinematic))

    data = all_subjects_filter(False, True)