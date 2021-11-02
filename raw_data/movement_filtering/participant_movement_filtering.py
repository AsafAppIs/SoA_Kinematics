import numpy as np
from raw_data.movement_filtering.filter import filtering
import raw_data.read_raw as read
from raw_data.normalization.hand_base_normalize import hand_base_normaliztion

# check wheather the trial meet three conditions
# 1) the y coordinate of the distal point is close at list 1cm to the base of the index finger
# 2) the distal point return to 2.5cm distance from init location
# 3) after that, it doesn't get 3.5cm away from it
def proper_movement_filter(trial):    
    # first condition. 28 for base & 34 for distal
    distal_pass = trial[(trial.iloc[:,28] + .025 > trial.iloc[:,34]) |  (trial.iloc[:,31] > trial.iloc[:,34])]
    if distal_pass.empty:
        return True
    
    # second & third conditions
    # get the getting back movement
    back_movement = trial.loc[distal_pass.index[0]:,:]
    
    # get only the distal coordinates
    back_movement = np.array(back_movement.iloc[:, 33:36])
    
    #get initial location of distal
    initial_location_distal = np.array(trial.iloc[0,33:36])
    
    # calculate distance from init
    distance = np.array([np. linalg.norm(back_movement[i] - initial_location_distal) for i in range(len(back_movement))])
   
    is_back = False
    for dis in distance:
        if dis < 0.035:
            is_back = True
        if dis > 0.045 and is_back: # the hand start to move far away again
            return True
    return not is_back
            
    
    
    

def movement_filtering(kinematics):
    # filter crazy trials
    num_of_unproper= filtering(kinematics, proper_movement_filter)
    
    
    return num_of_unproper 
 


if __name__ == "__main__":    
    experiment_num = 2
    participant_num = 1
    trial_num = 24
    
    kinematic = read.get_parsed_kinematic_data(experiment_num, participant_num)
    hand_base_normaliztion(kinematic)
    print(movement_filtering(kinematic))
    _, trial = kinematic[trial_num]
    proper_movement_filter(trial)
