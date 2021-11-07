# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import raw_data.configurations as cfg
import raw_data.read_raw as read
from raw_data.movement_filtering.leap_filtering import leap_filtering
from raw_data.movement_filtering.participant_movement_filtering import movement_filtering
from raw_data.normalization.hand_base_normalize import hand_base_normaliztion


def all_subjects_filter_and_save():
    colums_names = ['experiment_num', "participant_num", "total_num_of_trials", "no manipulation", "100ms", "200ms",
                    '300ms', '6d', '10d', '14d', 'num_of_fliped_hand', 'num_of_no_answer', 'num_of_crazy',
                    'num_of_jumps', 'num_of_unappropriate_length', 'num_of_small', 'num_of_not_flat', 'num_of_unproper']
    subjects_filtering_data = []
    for experiment, participant in cfg.all_participants:
        # extract data from csv
        data, extraction_filtered = read.get_integrated_information(experiment, participant)
        
        # normalize to hand base
        hand_base_normaliztion(data)
        
        # filtering
        filtered_trials = filter_subject(data)
        
        # extract trial type counts
        counts, is_legal = calculate_counts(data)

        line = (experiment, participant, len(data)) + tuple(counts) + tuple(extraction_filtered) + tuple(filtered_trials)
        subjects_filtering_data.append(list(line))

    df =  pd.DataFrame(subjects_filtering_data, columns = colums_names)
    
    df.to_csv(cfg.data_path+"/filtering_data.csv")

def all_subjects_filter(list_mode=True, normalize=False):
    subjects_lst = []
    good_subjects = []
    total_bad_trials = np.zeros((1,3))
    for experiment, participant in cfg.all_participants:
        # extract kinematic data from file
        subject_kinematics = read.get_integrated_information(experiment, participant)
        total_num_of_trials = len(subject_kinematics)
        # normalize
        if normalize:
            hand_base_normaliztion(subject_kinematics)
        bad_trials = filter_subject(subject_kinematics)
        # filter
        
        #total_bad_trials += filter_subject(subject_kinematics)
        print(f"Experiment {experiment} participant {participant:2}:", end=" ")
        print(f"total number: {total_num_of_trials}", end=" ")
        print(f"{bad_trials} which is {100*sum(bad_trials)/total_num_of_trials:.2f}%")
        
        # add to list
        subjects_lst.append(subject_kinematics)
        
        # print counts
        counts, is_legal = calculate_counts(subject_kinematics)
        print(counts, is_legal)
        if is_legal:
            good_subjects.append((experiment, participant))
    
    print(good_subjects)
    # if not on list_mode then aggregate and return
    if not list_mode:
        # concatenate all the movements from all participants together 
        whole_movement = np.concatenate(subjects_lst, axis=0)
        
        return whole_movement 
    
    #print(total_bad_trials)
    # if list_mode then return list
    return subjects_lst
 


# this function gets kinematic data list and a threshold
# and return whether there is enough trials for each type
def legal_counts(kinematics, threshold=24):
    # extract trial types
    types = [x[2] for x in kinematics]
    types, counts = np.unique(types,return_counts=True)
    is_legal = np.sum(counts[counts < threshold]) == 0
    
    return is_legal 


# this function gets kinematic data list and return the counts of each type
def calculate_counts(kinematics):
    # extract trial types
    types = [x[2] for x in kinematics]
    types, counts = np.unique(types,return_counts=True)
    is_legal = np.sum(counts[counts < 23]) == 0
    
    return counts, is_legal 


def filter_subject(kinematics):
    num_of_crazy, num_of_jumps, num_of_unappropriate_length = leap_filtering(kinematics)
    num_of_small, num_of_not_flat, num_of_unproper = movement_filtering(kinematics)
    return np.array((num_of_crazy, num_of_jumps, num_of_unappropriate_length, num_of_small, num_of_not_flat, num_of_unproper))

    
if __name__ == "__main__":    
    experiment_num = 2
    participant_num = 36
    trial_num = 24
    all_subjects_filter_and_save()

    #kinematic = read.get_integrated_information(experiment_num, participant_num)
    #hand_base_normaliztion(kinematic)
    #print(filter_subject(kinematic))
    #data = all_subjects_filter(False, True)