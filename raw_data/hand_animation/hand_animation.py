import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import mpl_toolkits.mplot3d.axes3d as p3
from raw_data.read_raw import get_parsed_kinematic_data
from raw_data.total_move.movement_normalization import movement_hand_normalization 
import raw_data.total_move.total_movement_extraction as tm_extract
 

experiment_num = 1
participant_num = 15
trial_num = 35

kinematic = get_parsed_kinematic_data(experiment_num, participant_num)
#_ = tm_extract.subject_total_movement(kinematic, True)
_, trial = kinematic[trial_num]
trial = trial.iloc[:,9:]
trial = np.array(trial, dtype=np.float)

trial = movement_hand_normalization(trial)

# find min max values
x_coordinates = np.arange(0,63,3)
y_coordinates = np.arange(1,63,3)
z_coordinates = np.arange(2,63,3)

trial_x = trial[:,x_coordinates]
trial_y = trial[:,y_coordinates]
trial_z = trial[:,z_coordinates] * -1

min_x = np.min(trial_x) - 0.05
min_y = np.min(trial_y) - 0.05
min_z = np.min(trial_z) - 0.05

max_x = np.max(trial_x) + 0.05
max_y = np.max(trial_y) + 0.05
max_z = np.max(trial_z) + 0.05


lines = [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16), (0,17), (17,18), (18,19), (19,20)]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.tick_params(axis='both', labelsize=0, length = 0)

def update(i):
    ax.clear()
    ax.set_xlim3d([min_x, max_x])
    ax.set_xlabel('X')
    
    ax.set_ylim3d([min_y, max_y])
    ax.set_ylabel('Y')
    
    ax.set_zlim3d([min_z, max_z])
    ax.set_zlabel('Z')
    for line in lines:        
        x1 = trial_x[i, line[0]]
        y1 = trial_y[i, line[0]]
        z1 = trial_z[i, line[0]]

        x2 = trial_x[i, line[1]]
        y2 = trial_y[i, line[1]]
        z2 = trial_z[i, line[1]]

        
        ax.plot([x1, x2],[y1, y2],[z1, z2], color='b', marker="o", markersize=3)
    time_text = ax.text(0.05, 0.95,.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    time_text.set_text(i)
ani = anim.FuncAnimation(fig, update, interval=100, frames=len(trial), repeat=True)


writer = anim.PillowWriter(fps=60)
#ani.save("regular_movement.gif", writer=writer)  
