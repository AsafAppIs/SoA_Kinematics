import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import raw_data.configurations as cfg
import raw_data.total_move.total_movement_extraction as tm_extract
import raw_data.read_raw as read

def plot_subject_trends(movement, participant=1, points_of_interest=[1,5,9,21]):
    
    
    
    #plot total movement trand
    fig, ax = plt.subplots(figsize=(12,6))
    
    window = sig.gaussian(15,7)
    window /= sum(window)
    
    for i in points_of_interest:
        data = np.convolve(movement[:,i-1], window)
        ax.plot(data[10:-10], label=i)
    ax.set_title(f"Total Movement trend {participant}", fontsize=20, pad=20)
    ax.set_xlabel("Trial num", fontsize=16, labelpad=20)
    ax.set_ylabel("Total Movement" , fontsize=16, labelpad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.grid(False)
    
for experiment, participant in cfg.all_participants:
    subject_kinematics = read.get_parsed_kinematic_data(experiment, participant)   
    movement, _ = tm_extract.subject_total_movement(subject_kinematics, normalize=False)
    plot_subject_trends(movement=movement, participant=participant)
