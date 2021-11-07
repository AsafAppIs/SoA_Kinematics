# -*- coding: utf-8 -*-
import raw_data.configurations as cfg
import pandas as pd
import os

# this function gets experiment number and index 
# and returns the right path to the exact participant
def create_path(experiment_num, participant_num):
    # input sanity check
    assert experiment_num in [1,2], 'Experiment number has to be 1 or 2'
    
    #create path to the right experiment
    experiment_path = cfg.first_data if experiment_num == 1 else cfg.second_data
    experiment_path = os.path.join(cfg.data_path, experiment_path)
    
    #create path to the right participat
    participant = "participant" + str(participant_num)
    assert participant in os.listdir(experiment_path), f"participant {participant_num} doesn't exist" 
    path = os.path.join(cfg.data_path, experiment_path, participant)
    
    return path



# This function gets experiment number and index 
# and returns a specific kinematics,answers and trials dataframes
def read_raw_data(experiment_num, participant_num):
    # get path to the participant data
    path = create_path(experiment_num, participant_num)
    
    kinematic_data = pd.read_csv(os.path.join(path, "Hands.csv"), header=None)
    answers = pd.read_csv(os.path.join(path, "Answers.csv"), header=None)
    trials = pd.read_csv(os.path.join(path, "Experiment.csv"), header=None)
    
    # for some reason there is an empty column in the end of the kinematic df
    # so we deleting it
    kinematic_data.drop(72, axis=1, inplace=True)
    
    return kinematic_data, answers, trials



# This function gets answers df
# its filter out trials with on answer
# and returns df (trials_num, response)
# Also, the function return the number of trials with no answer  
# 1 - SoA
# 0 - no SoA
def extract_answers(answers):
    # filter only SoA answers
    SoA_answers = answers[answers.iloc[:,9] == 1]
    
    #delete no answer rows
    num_of_no_answer = SoA_answers.iloc[:,4].isna().sum()
    SoA_answers = SoA_answers[SoA_answers.iloc[:,4].notnull()]
    
    # 'left' -> 0 & 'right' - > 1
    SoA_answers.iloc[:,4] = (SoA_answers.iloc[:,4] == 'Right').astype('int8')
    
    # take only trial number(6) and answer (4)
    SoA_answers = SoA_answers.iloc[:,[6,4]]
    
    #rename columns
    SoA_answers = SoA_answers.rename(columns={6:"trial number", 4:"SoA answer"})
    
    return SoA_answers, num_of_no_answer



# This function gets a trials df
# and returns df (trials_num, trial_type)
# 0 - no manipulation
# 1 - 3: temporal manipulation
# 4 - 6: spatial manipulation
# 7 - 9: anatomical manipulation 
def extract_trial_type(trials, anatomical_data=False):
    # dictionary that map every manipulation to its representation
    label_dict = {0:0, .1:1, .2:2, .3:3, 6:4, 10:5, 14:6, 3:7, 4:8, 5:9}
    
    # delete trials with anatomical manipulation
    if not anatomical_data:
        trials = trials[trials.iloc[:,7] == 0]
        
    # calculate manipulation representation
    trials['manipulation type'] = trials.iloc[:, 4:8].sum(axis=1)
    trials['manipulation type'] = trials['manipulation type'].map(label_dict)
    
    # create output dataframe
    new_trials = trials.iloc[:,[1,9]]
    
    #rename column
    new_trials = new_trials.rename(columns={1:"trial number"})
    
    return new_trials



# this function gets list of kinematic trials
# and filter trials where the hand was flipped
# the function return list without the fliiped trials 
# and the num of trials that were filtered
def filter_flipped_hand(trials):
    # take only trials where the the num of unflipped frames 
    # equal to length of the trial
    filtered_trials = [x for x in trials if x[1].iloc[:,8].sum() == x[1].shape[0]]
    
    # calculate number of filterd out trials
    num_of_fliped_hand = len(trials) - len(filtered_trials)
    
    return filtered_trials, num_of_fliped_hand



# this function gets groupby object
# and return a list of tuples (trial_num, trial_dataframe)
def convert_kinematics_to_list(group_kinematics):
    kinematic_list = []
    for name, df in group_kinematics:
        kinematic_list.append((name,df))
    return kinematic_list



# This function gets a kinematic df
# and returns list of tuples (trial_num, trial_dataframe)
# also, this function filter out trials when the hand fliped
def split_kinematic_to_trials(kinematics):
    # filter all the unnecessary frames, keep only the movement part
    # the movement part is markered by:
    # "ShowingHands" in the 5 col
    # and "Execution" in the 6 col
    new_kinematics = kinematics[(kinematics.iloc[:,5] == "ShowingHands") & (kinematics.iloc[:,6] == "Execution")]
    
    # groupby trial number (2)
    group_kinematics = new_kinematics.groupby(new_kinematics.columns[2])
    
    #convert groupby object to list of (trial_num, trial_dataframe)
    kinematic_list = convert_kinematics_to_list(group_kinematics)
    
    # filter trials with flipped hand 
    kinematic_list, num_of_fliped_hand = filter_flipped_hand(kinematic_list)

    return kinematic_list, num_of_fliped_hand


# This function gets experiment number and index 
# and returns list of tuples (trial_num, trial_dataframe)
def get_parsed_kinematic_data(experiment_num, participant_num):
    k,_,_ = read_raw_data(experiment_num, participant_num)
    df, _ = split_kinematic_to_trials(k)
    
    return df

# This function gets experiment number and index 
# and returns list of tuples (trial_num, full_trial_dataframe) 
# and number of data filtered trials (flip hand, no answer)
def get_integrated_information(experiment_num, participant_num):
    # get raw data
    kinematics, answers, trials = read_raw_data(experiment_num, participant_num)
    
    # split kinematics to trials
    kinematics, num_of_fliped_hand = split_kinematic_to_trials(kinematics)
    
    # extract trial type
    trial_type = extract_trial_type(trials)
    
    # extract answers
    answers, num_of_no_answer = extract_answers(answers)
    
    # integrate
    # define list that will contain indices to delete
    to_delete = []
    
    for i, (idx, df) in enumerate(kinematics):
        # if there is no answer or if trial is unavailale, we wil remove it later
        if idx not in trial_type['trial number'].values or idx not in answers['trial number'].values:
            to_delete.append(idx)
            continue
        
        # extract current trial type
        current_trial_type = trial_type[trial_type['trial number'] == idx]['manipulation type'].iloc[0]
        
        # extract current answer
        current_answer = answers[answers['trial number'] == idx]['SoA answer'].iloc[0]
        
        # edit the data list
        kinematics[i] = (idx, df, current_trial_type, current_answer)
    
    # delete partial trials
    kinematics = [trial for trial in kinematics if trial[0] not in to_delete]
    
    return kinematics, (num_of_fliped_hand, num_of_no_answer)
        
    
