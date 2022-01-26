import numpy as np
import raw_data.configurations as cfg
from raw_data.read_raw import get_integrated_information
from raw_data.normalization.hand_base_normalize import hand_base_normaliztion
from raw_data.movement_filtering.filtering_interface import filter_subject, legal_counts
from raw_data.intrapolating.intrapolating import subject_to_kinematic_representation

# This function gets experiment number and index 
# extract this participant, normalize thedata
# filter bad trials and extrapolate the length to 120
# transform each trial into line kinematic representation
# (to be done) then, it returns a ndarray in which each line represents a trial
def preprocess_subject(experiment_num, participant_num):
    # extract data from csv
    data, extraction_filtered = get_integrated_information(experiment_num, participant_num)
    
    # normalize to hand base
    #hand_base_normaliztion(data)
    
    # filtering
    filtered_trials = filter_subject(data)
    
    # check whether the amount of count is legal, if not return -1
    is_legal = legal_counts(data)
    if not is_legal:
        return -1
    
    # transform to kinematic representation 
    kinematic_representation = subject_to_kinematic_representation(data)
    
    return kinematic_representation


if __name__ == "__mai__":
    idx_counter = 1
    for experiment, participant in cfg.all_participants:
        kinematic_representation = preprocess_subject(experiment, participant)
        if kinematic_representation is -1:
            print(f"Experiment {experiment} participant {participant} don't have enough trials")
            continue
        print(f"Experiment {experiment} participant {participant} -> participant{idx_counter}")
        path = cfg.kinematic_path+"participant"+str(idx_counter)+".csv"
        np.savetxt(path, kinematic_representation, delimiter=',')
        idx_counter += 1
        
