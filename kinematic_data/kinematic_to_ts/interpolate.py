import numpy as np
from scipy.interpolate import interp1d
import kinematic_data.configurations as cfg


# this function gets a timeseries and interpolate it to the appropriate number of points
# as defined in the configurations file (optional)
def interpolate(ts, length=cfg.ts_length, interpolate_type='cubic'):
    # define intrapolation function
    x = np.arange(len(ts))
    f = interp1d(x,ts, kind=interpolate_type)
    
    # define new x in order 
    new_x = np.linspace(0, len(ts)-1, length)
        
    # intrapolate 
    new_ts = f(new_x)
    
    return new_ts
    