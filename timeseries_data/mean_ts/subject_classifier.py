import numpy as np
import timeseries_data.configurations as cfg
from timeseries_data.mean_ts.trial_classifier import trial_classfier_creator
from timeseries_data.read_timeseries import read_subject
import timeseries_data.util.util as util

# this function gets a subject timeseries data and classifier function
# and return two lists of indices that represent the rows of the two classes
def subject_classifier(subject_data, classifier):
    # define list to contain the indices for both classes and the unclassified
    idx_lst = [[],[],[]]
    
    # iterate over the trials
    for i, line in enumerate(subject_data):
        idx = classifier(line)
        idx_lst[idx].append(i)
     
    # return only the first & seconds lists of indices 
    return idx_lst[:2]



# this function gets subject timeseries data and coniguration for classifier function
# and return mean timesseries for both conditions of the classifier
def conditional_mean_ts(subject_data, config=False, idx=0, class_dict=False):
    # trnasfrom the subject data into ndarry
    subject_data = np.array(subject_data)
    
    # define classifier function
    classifier = trial_classfier_creator(config=config, idx=idx, class_dict=class_dict)
    
    # extract the indices of the two trials
    first_class_idx, second_class_idx = subject_classifier(subject_data, classifier)
    
    # calculate means
    first_class_mean = np.mean(subject_data[first_class_idx], axis=0)
    second_class_mean = np.mean(subject_data[second_class_idx], axis=0)
    
    # calculate std
    first_class_std = np.std(subject_data[first_class_idx], axis=0)
    second_class_std = np.std(subject_data[second_class_idx], axis=0)
    
    #calculate ste
    first_class_ste = first_class_std / np.sqrt(len(first_class_idx))
    second_class_ste = second_class_std / np.sqrt(len(second_class_idx))


    # remove header
    first_class_mean = first_class_mean[cfg.header_size:]
    first_class_ste = first_class_std[cfg.header_size:]
    second_class_mean = second_class_mean[cfg.header_size:]
    second_class_ste = second_class_std[cfg.header_size:]
    
    return first_class_mean, first_class_ste, second_class_mean, second_class_ste



# this function gets coniguration for classifier function
# and return mean timesseries for both conditions of the classifier for all the participants
def all_mean_ts(config=False, idx=0, class_dict=False):
    # define classifier function
    classifier = trial_classfier_creator(config=config, idx=idx, class_dict=class_dict)
    
    # define container for both classes
    first_lst = []
    second_lst = []
    
    for i in range(cfg.num_of_participants):
        print(i)
        data = read_subject(i+1)
        data = util.subject_filter_medial(data)
        
        # extract the indices of the two trials
        first_class_idx, second_class_idx = subject_classifier(data, classifier)
        
        # add trial to classes containers
        first_lst.append(data[first_class_idx])
        second_lst.append(data[second_class_idx])
        
    # concatenate
    first_class = np.concatenate(first_lst, axis=0)
    second_class = np.concatenate(second_lst, axis=0)
    
    # calculate means
    first_class_mean = np.mean(first_class, axis=0)
    second_class_mean = np.mean(second_class, axis=0)
    
    # calculate std
    first_class_std = np.std(first_class, axis=0)
    second_class_std = np.std(second_class, axis=0)
    
    #calculate ste
    first_class_ste = first_class_std / np.sqrt(len(first_class))
    second_class_ste = second_class_std / np.sqrt(len(second_class))


    # remove header
    first_class_mean = first_class_mean[cfg.header_size:]
    first_class_ste = first_class_std[cfg.header_size:]
    second_class_mean = second_class_mean[cfg.header_size:]
    second_class_ste = second_class_std[cfg.header_size:]
    
    return first_class_mean, first_class_ste, second_class_mean, second_class_ste
        

