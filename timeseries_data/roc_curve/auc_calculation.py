import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import random 
import timeseries_data.configurations as cfg
from timeseries_data.mean_ts.subject_classifier import subject_classifier
from timeseries_data.mean_ts.trial_classifier import trial_classfier_creator


def auc_calculator(data, index1, index2, measure_idx):
    # if one class is too small, return nan
    if (len(index1) < 6) or (len(index2) < 6):
        return np.nan
    
    # calclate real measure index
    actual_idx = cfg.header_size+measure_idx
    
    new_data = np.array([(data[i, actual_idx], 1) if (i in index1) else (data[i, actual_idx], 0) if (i in index2) else -1 
                         for i in range(len(data))], dtype=np.object)
    new_data = np.array([x for x in new_data if x is not -1], dtype=np.float32)
    
    
    weight1 = len(index1) / len(new_data)
    weight2 = len(index2) / len(new_data)
    
    index1_array = new_data[:, 1]
    weight_array = np.array([weight1 if index1_array[i]else weight2 for i in range(len(new_data))])
    #weight_array = np.array([1 if index1_array[i] else 1 for i in range(len(new_data))])

    measure_data = new_data[:, 0]
    
    auc = roc_auc_score(index1_array, measure_data, sample_weight=weight_array)   
    
    return auc


# this function calculate the auc of large number of permutation 
# of trials from some split
def permutation_auc(permutations = 1000, config=False, idx=0, class_dict=False):
    random.seed(10)
    results = []
    for i in range(cfg.num_of_participants):
        data = pd.read_csv(cfg.special_feature_path + "participant" + str(i+1) + ".csv", header=None)
        data = np.array(data)
        
        # create calculator
        classifier = trial_classfier_creator(config=config, idx=idx, class_dict=class_dict)
        # calculate indices
        first_class_idx, second_class_idx = subject_classifier(data, classifier)
        all_idx = first_class_idx + second_class_idx 
        split = int(len(all_idx)/2)
        for k in range(permutations):
            random.shuffle(all_idx)
            idx1 = all_idx[:split]
            idx2 = all_idx[split:]
            auc = [auc_calculator(data, idx1, idx2, i) for i in range(3)]
            results.append(auc)
            
    return np.array(results)


# this function gets subject number and return the auc under different data splits
def subject_auc(subject_num):
    results = [subject_num]
    # read data
    data = pd.read_csv(cfg.special_feature_path + "participant" + str(subject_num) + ".csv", header=None)
    data = np.array(data)
    
    for i, conf in enumerate(cfg.class_configurations[:14]):
        # create calculator
        if isinstance(conf, tuple):
            classifier = trial_classfier_creator(idx=conf[1], class_dict=conf[2])
        else:
            classifier = trial_classfier_creator(config=conf)

        # calculate indices
        first_class_idx, second_class_idx = subject_classifier(data, classifier)
        
        # calculate auc for all kinematic measures
        auc = [auc_calculator(data, first_class_idx, second_class_idx, i) for i in range(3)]
        
        # add this split results to results list
        results.extend(auc)
    
    return results

# this function calculate the auc of all the participants under all splits
def all_auc():
    results = []
    for i in range(cfg.num_of_participants):
        results.append(subject_auc(i+1))
    
    df = pd.DataFrame(results)
    df.to_csv(cfg.auc_path + "participants.csv", header=None, index=None)
    
    return np.array(results)

