import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# this function gets kinematic data frame 
# and returns only the coordinates of the distal and medial points of the index finger
def only_relevant_data_points(kinematic_df):
    return kinematic_df.iloc[:, 30:36]



# this functions gets kinematic data frame (pandas dataframe, only distal&medial) and 
# intrapolate the data into 120 data point (numpy array)
# the function return flat ndarray 120*6=720
def trial_intraplote(kinematic_df):
    # define ndarray that will contain the kinematic data in the end
    kinematic_array = np.zeros((6,120))
    
    # iterate
    for i in range(6):
        # define intrapolation function
        y = kinematic_df.iloc[:,i]
        x = np.arange(len(y))
        f = interp1d(x,y, kind='cubic')
        
        # define new x in order 
        new_x = np.linspace(0, len(y)-1, 120)
        
        # intrapolate 120 points
        new_y = f(new_x)
        
        kinematic_array[i] = new_y
    
    # return flatten version of the array
    return kinematic_array.flatten()


# this function transfrom trial from full df represention into
# "kinematic representation" which is a line containing:
# trial num - 1
# trial type - 1
# answer = 1
# location kinematic data of 3 coordinates in the distal&medial along 120 frames - 2*3*120 = 720
# total - 723
# this function returns 1d ndarray in size of 723
def trial_to_kinematic_representation(trial_data):
    # create the ndarray
    kinematic_representation = np.zeros(723)
    
    # assign trial num
    kinematic_representation[0] = int(trial_data[0])
    
    # assign trial type
    kinematic_representation[1] = int(trial_data[2])
    
    # assign answer
    kinematic_representation[2] = int(trial_data[3])
    
    # assign kinematics
    kinematics = only_relevant_data_points(trial_data[1])
    kinematic_representation[3:] = trial_intraplote(kinematics)
    
    return kinematic_representation


# this function gets subject data (list of tuples)
# and return ndarray (trialnum X 723) of kinematic representation
def subject_to_kinematic_representation(subject_data):
     # create the ndarray
    subject_kinematic_representation = np.zeros((len(subject_data),723))
    
    for i in range(len(subject_data)):
        subject_kinematic_representation[i] = trial_to_kinematic_representation(subject_data[i])
        
    return subject_kinematic_representation

        