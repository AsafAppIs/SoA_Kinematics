# this script use to filter out trials in which there is a problem with the leap record
# we have two criteria for that:
# 1) crazy hand: when the hand jump around -> located by total movement bigger then
# 2) jump: the hand suddenly jump away from it place -> located by frame in which the total hand mocement is bigger then 12cm

import numpy as np
from raw_data.movement_filtering.filter import filtering
from raw_data.total_move.total_movement_extraction import trial_total_movement
import raw_data.read_raw as read
from raw_data.normalization.hand_base_normalize import hand_base_normaliztion


def crazy_movement_filtering(trial):
    total_movement = trial_total_movement(trial)
    return np.any(total_movement[total_movement > 1.5])


def jump_filtering(trial):
    kinematic_df = trial.iloc[:,9:]
    kinematic_array = np.array(kinematic_df, dtype=np.float)
    
    kinematic_diff = np.diff(kinematic_array, axis=0)
    # calculte the euclidian distance made
    # split to points, square, sum, sqrt
    kinematic_points = np.split(kinematic_diff, indices_or_sections=21, axis=1)
    for i in range(len(kinematic_points)):
        kinematic_points[i] = kinematic_points[i] ** 2
        kinematic_points[i] = np.sum(kinematic_points[i], axis=1)
        kinematic_points[i] = np.sqrt(kinematic_points[i])
        
    sum_movement = np.sum(kinematic_points, axis=0)
    return np.any(sum_movement[sum_movement>0.15])


# filter trials with length outside the range of 119-121
def trial_length_filter(trial):
    return not(118 < len(trial) < 122)




def leap_filtering(kinematics):
    # filter crazy trials
    num_of_crazy = filtering(kinematics, crazy_movement_filtering)
    
    # filter trials with jumps
    num_of_jumps = filtering(kinematics, jump_filtering)
    
    # filter trials with unappropriate length
    num_of_unappropriate_length = filtering(kinematics, trial_length_filter)
    
    return num_of_crazy, num_of_jumps, num_of_unappropriate_length



if __name__ == "__main__":    
    kinematic_data = read.get_parsed_kinematic_data(2, 36)
    hand_base_normaliztion(kinematic_data )
    _, trial = kinematic_data[75]
    #jump_filtering(trial)
    num_of_crazy, num_of_jumps = leap_filtering(kinematic_data)