import numpy as np
import raw_data.total_move.total_movement_extraction as tm_extract

# this function gets total movement data and 
# return ndarray (4*21) of mean, std, min, max
def total_movement_stats_calculation(movement):
    mean_movement = np.mean(movement, axis=0).reshape((1,-1))
    std_movement = np.std(movement, axis=0).reshape((1,-1))
    min_movement = np.min(movement, axis=0).reshape((1,-1))
    max_movement = np.max(movement, axis=0).reshape((1,-1))
    
    # aggregate
    movement_stats = np.concatenate([mean_movement, std_movement, min_movement, max_movement], axis=0)

    return movement_stats 



# this function gets total movement data and print
# the mean, std, min, max of each point total movement
def total_movement_stats_report(movement):
    # extract total movement data
    movement_stats = total_movement_stats_calculation(movement)
    # print points report
    for i in range(21):
        print(f"point number {i+1:2}:  min: {movement_stats[2,i] :.3f} max: {movement_stats[3,i] :.3f} mean: {movement_stats[0,i] :.3f} std: {movement_stats[1,i] :.3f}")
