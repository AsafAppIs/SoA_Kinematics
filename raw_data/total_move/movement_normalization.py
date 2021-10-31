import numpy as np
import raw_data.configurations as cfg

# this function gets a movement of a trial 
def movement_hand_normalization(kinematic_data):
    # split the kinematic data to the different points
    kinematic_points = np.split(kinematic_data, indices_or_sections=21, axis=1)
    # normalize the movement to the base of the hand (point 0)
    for i in range(20,-1,-1):
        kinematic_points[i] = kinematic_points[i] - kinematic_points[0]
    
    # aggregate the points together again
    kinematic_data = np.concatenate(kinematic_points, axis=1)
    
    return kinematic_data
    


