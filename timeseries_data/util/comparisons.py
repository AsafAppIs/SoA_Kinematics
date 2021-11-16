import numpy as np

# this function calculate the correlation between to timeseries
def correlation_ts(first_ts, second_ts):
    return np.corrcoef(first_ts, second_ts)[0,1]