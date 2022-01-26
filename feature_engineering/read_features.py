import pandas as pd
import feature_engineering.configurations as cfg

def read_features(participant_num, mode="clean"):
    if mode == "full":
        feature_path = cfg.full_feature_path
    elif mode == "clean":
        feature_path = cfg.full_clean_feature_path
        
    path = feature_path + "participant" + str(participant_num) + ".csv"
    
    # read the data
    data = pd.read_csv(path)
        
    return data 
