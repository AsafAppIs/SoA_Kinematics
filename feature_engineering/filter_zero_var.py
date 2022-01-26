import pandas as pd
import numpy as np
import feature_engineering.configurations as cfg
from feature_engineering.read_features import read_features 


def filter_zero():
    idx_to_filter = np.ones(8660)
    idx_to_filter = np.array(idx_to_filter, dtype=bool)
    
    for i in range(cfg.num_of_participants):
        print(f"analyzing participant {i+1}")
        data = read_features(i+1)
        zero_var = data.std(axis=0)
        zero_var = (zero_var == 0)
        zero_var = np.array(zero_var)
        
        idx_to_filter = (idx_to_filter & zero_var)
        
    print(f"filtering {np.sum(idx_to_filter)} unrelevant features")
        
    for i in range(cfg.num_of_participants):
        print(f"rewriting participant {i+1}")
        data = read_features(i+1)
        data = data[data.columns[~idx_to_filter]]
        
        data.to_csv(cfg.full_clean_feature_path+"participant"+str(i+1)+".csv", index=False)
        
        
filter_zero()