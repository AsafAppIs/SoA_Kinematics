import numpy as np
import pandas as pd
import raw_data.configurations as cfg
from raw_data.read_raw import get_parsed_kinematic_data
# this function gets a movement of a trial 
def movement_hand_normalization(kinematic_data):
    
    kinematic_data = np.array(kinematic_data)
    # split the kinematic data to the different points
    kinematic_points = np.split(kinematic_data, indices_or_sections=21, axis=1)
    # normalize the movement to the base of the hand (point 0)
    for i in range(20,-1,-1):
        kinematic_points[i] = kinematic_points[i] - kinematic_points[0]
    
    # aggregate the points together again
    kinematic_data = np.concatenate(kinematic_points, axis=1)
    return kinematic_data
 

# this function gets kinemtaic data 
# and normalize the data according to the hand base
def hand_base_normaliztion(subject_kinematics):
    for i, (idx, trialdf, *args)in enumerate(subject_kinematics):
        
        normalized_data = movement_hand_normalization(trialdf.iloc[:,9:])
        subject_kinematics[i] = (idx, pd.DataFrame(np.concatenate((trialdf.iloc[:,:9], normalized_data), axis=1)),*args )
        
        
