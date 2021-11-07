import numpy as np
from raw_data.movement_filtering.filter import filtering
import raw_data.read_raw as read
from raw_data.normalization.hand_base_normalize import hand_base_normaliztion

# check wheather the trial meet three conditions
# 1) the y coordinate of the distal point is close at list 1cm to the base of the index finger
# 2) the distal point return to 2.5cm distance from init location
# 3) after that, it doesn't get 3.5cm away from it
def small_movement_filter(trial):    
    # first condition. 28 for base & 34 for distal
    distal_pass = trial[(trial.iloc[:,28] + .025 > trial.iloc[:,34]) |  (trial.iloc[:,31] > trial.iloc[:,34])]
    if distal_pass.empty:
        return True
    return False

def proper_end_filter(trial):
    # second & third conditions
    # take the last 20 frames
    back_movement = trial.iloc[-20:,:]
    
    # get only the distal coordinates
    back_movement = np.array(back_movement.iloc[:, 33:36])
    
    #get initial location of distal
    initial_location_distal = np.array(trial.iloc[0,33:36])
    
    # calculate distance from init
    distance = np.array([np. linalg.norm(back_movement[i] - initial_location_distal) for i in range(len(back_movement))])
    
    # check whether it get ba
    is_back = False
    for dis in distance:
        if dis < 0.025:
            is_back = True
        if dis > 0.035 and is_back: # the hand start to move far away again
            return True
    return not is_back
            

# throw trials were the initial distal point is higher than the finger base in 2cm
def flat_hand_filter(trial):
    return trial.iloc[0,35] < trial.iloc[0,29] - 0.02
    


def proper_end_filter_y(trial):
    # second & third conditions
    # take the last 20 frames
    back_movement = trial.iloc[-20:,:]
    
    # get only the distal coordinates
    back_movement = np.array(back_movement.iloc[:, 34])
    
    #get initial location of distal
    initial_location_distal = np.array(trial.iloc[0,34])
    
    # calculate distance from init
    distance = np.array([np. linalg.norm(back_movement[i] - initial_location_distal) for i in range(len(back_movement))])
    
    # check whether it get ba
    is_back = False
    for dis in distance:
        if dis < 0.025:
            is_back = True
        if dis > 0.035 and is_back: # the hand start to move far away again
            return True
    return not is_back
    
    

def movement_filtering(kinematics):
    # filter small trials
    num_of_small = filtering(kinematics, small_movement_filter)
    
    num_of_not_flat = filtering(kinematics, flat_hand_filter)
    #filter trials with unproper ending
    num_of_unproper = filtering(kinematics, proper_end_filter_y)
    
    return num_of_small, num_of_not_flat, num_of_unproper
 


if __name__ == "__main__":    
    experiment_num = 2
    participant_num = 42
    trial_num = 188
    
    kinematic = read.get_integrated_information(experiment_num, participant_num)
    hand_base_normaliztion(kinematic)
    print(movement_filtering(kinematic))
    _, trial = kinematic[trial_num]
    proper_end_filter(trial)
