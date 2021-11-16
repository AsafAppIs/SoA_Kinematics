import numpy as np

import kinematic_data.configurations as cfg
from kinematic_data.kinematic_to_ts.interpolate import interpolate


# this function gets list of timeseries, calculate the euclidian combination
# and return it
def euclidian_combination_ts(derivative_ts_lst):
    # cast the list to ndarray in order to calculate
    derivative_ts_array = np.array(derivative_ts_lst)
    
    # square, sum and sqrt
    derivative_ts_array = derivative_ts_array  ** 2
    combination_ts = np.sum(derivative_ts_array, axis=0)
    combination_ts = np.sqrt(combination_ts)
    
    return combination_ts 
    



# this function gets timeseries of one coordinate
# and return timeseries of derivative
def coordinate_derivative(coordinate_ts):
    derivative = np.diff(coordinate_ts)
    derivative /= cfg.time_interval_between_frames
    return derivative



# this function gets timeseries of sensor coordinates
# calculate the derivative of each one of them and the euclidian combination as well
# the function returns ndarray of one coordinate derivative timeseries and ndarray of euclidian combination of the sensor
def sensor_derivative_timeseries(sensor_ts):
    # define list containers for coordinate derivatives
    derivative_ts_lst = []
    
    # iterate over the timeseries and calculate the derivative
    for i in range(len(sensor_ts)):
        derivative_ts = coordinate_derivative(sensor_ts[i])
        derivative_ts = interpolate(derivative_ts)
        derivative_ts_lst.append(derivative_ts)
        
    # calculate euclidian combination
    combintions_ts = euclidian_combination_ts(derivative_ts_lst)
    
    # concat all of the derivative timeseries in the list
    derivative_ts = np.concatenate(derivative_ts_lst, axis=0)

    
    return derivative_ts, combintions_ts


# this function gets many kinematic timseries
# split in to different sensors and calculate derivative for each 
# and calculate a euclidian combination of each sensor
# the function returns ndarray of one coordinate derivative timeseries and ndarray of euclidian combinations
def calculate_derivative_timeseries(original_ts):
    # define list containers for both one coordinate derivative and euclidian combinations derivative
    derivative_ts_lst = []
    combintions_lst = []
    
    # split the ts to rows
    original_ts = np.reshape(original_ts, (cfg.num_of_original_ts, cfg.ts_length))
    
    # iterate over each group of sensors
    for i in range(0, len(original_ts), cfg.num_of_coordinates):
        # take the current sensor coordinates
        sensor_ts = original_ts[i:i+cfg.num_of_coordinates]
        
        # calculate derivative timeseries
        derivative_ts, combintions_ts = sensor_derivative_timeseries(sensor_ts)
        
        # add timeseries to containers
        derivative_ts_lst.append(derivative_ts)
        combintions_lst.append(combintions_ts)
        
    # concat all of the derivative timeseries in the lists
    derivative_ts = np.concatenate(derivative_ts_lst, axis=0)
    combintions_ts = np.concatenate(combintions_lst, axis=0)
    
    return derivative_ts, combintions_ts