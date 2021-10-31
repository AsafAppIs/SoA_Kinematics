import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

from raw_data.total_move.movement_normalization import movement_hand_normalization 

# this function get a trial kinematic data frame
# and returns a vector that contain the total movement 
# done by all the 21 points recorded by leap

coor_dict = {0:'x', 1:'y', 2:'z'}

def subject_location(subject_kinematics,participant=1, normalize=False, points_of_interest=[1,5,9,21]):
    movement = np.zeros(shape=(len(subject_kinematics), 63))
    
    # iterate over trials and extract movement data
    for i, trial in enumerate(subject_kinematics):
        movement[i] = trial_location(trial[1], normalize=normalize)
    
    #plot total movement trand
    fig, ax = plt.subplots(figsize=(12,18), nrows=3)
    
    window = sig.gaussian(15,7)
    window /= sum(window)
    for num in [0,1,2]:
        coordinate = coor_dict[num]
        for i in points_of_interest:
            idx = (i-1)*3+num
            data = np.convolve(movement[:,idx], window)
            ax[num].plot(data[10:-10], label=f"{i} coordinate")
        ax[num].set_title(f"Total location trend {participant} {coordinate} coordinate", fontsize=20, pad=20)
        ax[num].set_xlabel("Trial num", fontsize=16, labelpad=20)
        ax[num].set_ylabel("mean location" , fontsize=16, labelpad=20)
        ax[num].spines['top'].set_visible(False)
        ax[num].spines['right'].set_visible(False)
        ax[num].spines['bottom'].set_color('black')
        ax[num].spines['left'].set_color('black')
        ax[num].spines['bottom'].set_linewidth(0.5)
        ax[num].spines['left'].set_linewidth(0.5)
        ax[num].grid(False)





def trial_location(trial_df, normalize=False):
    # take only the kinematic info from the file and convert it to numpy array
    kinematic_df = trial_df.iloc[:,9:]
    kinematic_array = np.array(kinematic_df, dtype=np.float)

    # if normalize flag is on: normalize the locations to hand base
    if normalize:
        kinematic_array = movement_hand_normalization(kinematic_array)
    

    mean_location = np.mean(kinematic_array, axis=0)
    
    return np.array(mean_location)


