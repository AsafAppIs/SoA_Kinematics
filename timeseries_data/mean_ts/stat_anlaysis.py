import numpy as np
from scipy import stats

import timeseries_data.mean_ts.ts_statistics as ts_stat
from timeseries_data.mean_ts.subject_classifier import subject_classifier
from timeseries_data.mean_ts.trial_classifier import trial_classfier_creator
from timeseries_data.read_timeseries import read_subject
from soa_stats.soa_stats import subject_agency_statistics
import timeseries_data.util.util as util
import timeseries_data.configurations as cfg
import timeseries_data.util.soa_equal as equal



# this function gets subject data, function thats computes specific statistic property of trials 
# and configuration of trials classifier
# the function will split the trials into two classes, compute the statistic for each trial in both groups
# and return the results in two seperate lists
def subject_statisitical_analysis(subject_data, statistics_function, config=False, idx=0, class_dict=False):
    # trnasfrom the subject data into ndarry
    subject_data = np.array(subject_data)
    
    # define classifier function
    classifier = trial_classfier_creator(config=config, idx=idx, class_dict=class_dict)

    # extract the indices of the two trials
    first_class_idx, second_class_idx = subject_classifier(subject_data, classifier)
    
    # equalize number of trials in each group if the soa condition was choose
    if config == "soa":
        first_class_idx, second_class_idx = equal.soa_equalizer_down(subject_data, first_class_idx, second_class_idx)
    
    # compute statistics
    first_class = [statistics_function(trial) for trial in subject_data[first_class_idx]]
    second_class = [statistics_function(trial) for trial in subject_data[second_class_idx]]
    
    return first_class, second_class



# this function calculate and return the mean a specific statistic of each participant in both classes
# and return two list in equal size of the paired statistics
def all_mean_stat_comparison(statistics_function, config=False, idx=0, class_dict=False, range_limit=cfg.num_of_participants, test=False):
    mean_results = np.zeros((range_limit, 2))
    for i in range(range_limit):
        data = read_subject(i+1, test=test)
        if not test:
            data = util.subject_filter_medial(data)
        
        # calculate the statistic for both classes
        first_class, second_class = subject_statisitical_analysis(data, statistics_function=statistics_function, 
                                                                config=config, idx=idx, class_dict=class_dict)
        
        # if there is not enough datapoints from each class, assign nan into array
        #if len(first_class) < 6 or len(second_class) < 6:
        #    mean_results[i,0] = np.nan
        #    mean_results[i,1] = np.nan
            
        # calculate the mean and add it into the results array
        mean_results[i,0] = np.mean(first_class)
        mean_results[i,1] = np.mean(second_class)
    
    # delete nan values
    mean_results = mean_results[~np.isnan(mean_results).any(axis=1)]
    return mean_results


# this function return the p value of the ttest and the mean difference
def calculate_stats(mean_results):
    _ ,pval = stats.ttest_rel(mean_results[:,0], mean_results[:,1])
    mean_diff = np.mean(mean_results[:,0] - mean_results[:,1])

    return pval, mean_diff


def subject_statistic_and_agency(range_subject=cfg.num_of_participants, test=False):
    # create a list to contain subjects data
    subjects_data = []
    for i in range(range_subject):
        print(i)
        # create list to contain subject data
        subject_data = []
        # read data
        data = read_subject(i+1, test=test)
        if not test:
            data = util.subject_filter_medial(data) 
        else:
            data = np.array(data)
        
        # calculate statistic difference for various data splits
        for j, conf in enumerate(cfg.class_configurations[:11]):
            # first calculation, 50:80 in the y axis- location
            # create statistics function
            ts_num = 1
            start=50
            end=80
            y_loc_statistics_function = ts_stat.part_trial_mean_creator(ts_num, start, end)
            
            # calculate the statistic
            if isinstance(conf, tuple):
                first_class, second_class = subject_statisitical_analysis(data, y_loc_statistics_function, idx=conf[1], class_dict=conf[2])
            else:
                first_class, second_class = subject_statisitical_analysis(data, y_loc_statistics_function, config=conf)
             
            # add difference
            subject_data.append(np.mean(first_class) - np.mean(second_class))
            
            
            # second calculation, 50:80 in the z axis location
            
            # create statistics function
            ts_num = 2
            start=50
            end=80
            z_loc_statistics_function = ts_stat.part_trial_mean_creator(ts_num, start, end)
            
            # calculate the statistic
            if isinstance(conf, tuple):
                first_class, second_class = subject_statisitical_analysis(data, z_loc_statistics_function, idx=conf[1], class_dict=conf[2])
            else:
                first_class, second_class = subject_statisitical_analysis(data, z_loc_statistics_function, config=conf)
             
            # add difference
            subject_data.append(np.mean(first_class) - np.mean(second_class))
            
            
            # third calculation, 40:70 in total acceleration timeseries
            
            # create statistics function
            ts_num = 10
            start=40
            end=70
            total_acc_statistics_function = ts_stat.part_trial_mean_creator(ts_num, start, end)
            
            # calculate the statistic
            if isinstance(conf, tuple):
                first_class, second_class = subject_statisitical_analysis(data, total_acc_statistics_function, idx=conf[1], class_dict=conf[2])
            else:
                first_class, second_class = subject_statisitical_analysis(data, total_acc_statistics_function, config=conf)
             
            # add difference
            subject_data.append(np.mean(first_class) - np.mean(second_class))
            
            
        # add agency statistics
        subject_data.extend(subject_agency_statistics(data))
        
        # add subject to the big list
        subjects_data.append(subject_data)
        
    return subjects_data

def statistical_tables(range_subject=cfg.num_of_participants, test=False, cfg_range=len(cfg.class_configurations)):
    # create list to contain results
    results = []
    
    for i, conf in enumerate(cfg.class_configurations[:cfg_range]):
        print(i)
        # create list to contain specific split results, created with the line name in it
        split_results = []
        
        # first calculation, 50:80 in the y axis location
        
        # create statistics function
        ts_num = 1
        start=50
        end=80
        y_loc_statistics_function = ts_stat.part_trial_mean_creator(ts_num, start, end)
        
        # calculate the statistic
        if isinstance(conf, tuple):
            mean_results = all_mean_stat_comparison(y_loc_statistics_function, idx=conf[1], class_dict=conf[2], range_limit=range_subject, test=test)
        else:
            mean_results = all_mean_stat_comparison(y_loc_statistics_function, config=conf, range_limit=range_subject, test=test)
        
        split_results.append(mean_results[:,0])
        split_results.append(mean_results[:,1])
        
        
        # second calculation, 50:80 in the z axis location
        
        # create statistics function
        ts_num = 2
        start=50
        end=80
        z_loc_statistics_function = ts_stat.part_trial_mean_creator(ts_num, start, end)
        
        # calculate the statistic
        if isinstance(conf, tuple):
            mean_results = all_mean_stat_comparison(z_loc_statistics_function, idx=conf[1], class_dict=conf[2], range_limit=range_subject, test=test)
        else:
            mean_results = all_mean_stat_comparison(z_loc_statistics_function, config=conf, range_limit=range_subject, test=test)
        
        split_results.append(mean_results[:,0])
        split_results.append(mean_results[:,1])
        
        # third calculation, 40:70 in total acceleration timeseries
        
        # create statistics function
        ts_num = 10
        start=40
        end=70
        total_acc_statistics_function = ts_stat.part_trial_mean_creator(ts_num, start, end)
        
        # calculate the statistic
        if isinstance(conf, tuple):
            mean_results = all_mean_stat_comparison(total_acc_statistics_function, idx=conf[1], class_dict=conf[2], range_limit=range_subject, test=test)
        else:
            mean_results = all_mean_stat_comparison(total_acc_statistics_function, config=conf, range_limit=range_subject, test=test)
        
        split_results.append(mean_results[:,0])
        split_results.append(mean_results[:,1])
        
        
        # fourth calculation, length of movement in y-loc ts
        ts_num = 1
        fun = ts_stat.derivative_peak_distance(ts_num)
        
        # calculate the statistic
        if isinstance(conf, tuple):
            mean_results = all_mean_stat_comparison(fun, idx=conf[1], class_dict=conf[2], range_limit=range_subject, test=test)
        else:
            mean_results = all_mean_stat_comparison(fun, config=conf, range_limit=range_subject, test=test)
        
        split_results.append(mean_results[:,0])
        split_results.append(mean_results[:,1])
        
        
        # fifth calculation, length of movement in y-loc ts
        ts_num = 2
        fun = ts_stat.derivative_peak_distance(ts_num)
        
        # calculate the statistic
        if isinstance(conf, tuple):
            mean_results = all_mean_stat_comparison(fun, idx=conf[1], class_dict=conf[2], range_limit=range_subject, test=test)
        else:
            mean_results = all_mean_stat_comparison(fun, config=conf, range_limit=range_subject, test=test)
        
        split_results.append(mean_results[:,0])
        split_results.append(mean_results[:,1])
        
        
        # add all to the results list
        results.append(np.array(split_results))
        
    return results



def statistical_tests(range_subject=cfg.num_of_participants, test=False, cfg_range=len(cfg.class_configurations)):
    # create list to contain results
    results = []
    
    for i, conf in enumerate(cfg.class_configurations[:cfg_range]):
        print(i)
        # create list to contain specific split results, created with the line name in it
        split_results = [cfg.class_configurations_names[i]]
        
        # first calculation, 50:80 in the y axis location
        
        # create statistics function
        ts_num = 1
        start=50
        end=80
        y_loc_statistics_function = ts_stat.part_trial_mean_creator(ts_num, start, end)
        
        # calculate the statistic
        if isinstance(conf, tuple):
            mean_results = all_mean_stat_comparison(y_loc_statistics_function, idx=conf[1], class_dict=conf[2], range_limit=range_subject, test=test)
        else:
            mean_results = all_mean_stat_comparison(y_loc_statistics_function, config=conf, range_limit=range_subject, test=test)
        
        # save results
        # save the number of participant in this comparison
        split_results.append(len(mean_results))
        # save the pvalue and the mean diff
        split_results.extend(calculate_stats(mean_results))
        
        
        # second calculation, 50:80 in the z axis location
        
        # create statistics function
        ts_num = 2
        start=50
        end=80
        z_loc_statistics_function = ts_stat.part_trial_mean_creator(ts_num, start, end)
        
        # calculate the statistic
        if isinstance(conf, tuple):
            mean_results = all_mean_stat_comparison(z_loc_statistics_function, idx=conf[1], class_dict=conf[2], range_limit=range_subject, test=test)
        else:
            mean_results = all_mean_stat_comparison(z_loc_statistics_function, config=conf, range_limit=range_subject, test=test)
        
        # save the pvalue and the mean diff
        split_results.extend(calculate_stats(mean_results))
        
        # third calculation, 40:70 in total acceleration timeseries
        
        # create statistics function
        ts_num = 10
        start=40
        end=70
        total_acc_statistics_function = ts_stat.part_trial_mean_creator(ts_num, start, end)
        
        # calculate the statistic
        if isinstance(conf, tuple):
            mean_results = all_mean_stat_comparison(total_acc_statistics_function, idx=conf[1], class_dict=conf[2], range_limit=range_subject, test=test)
        else:
            mean_results = all_mean_stat_comparison(total_acc_statistics_function, config=conf, range_limit=range_subject, test=test)
        # save the pvalue and the mean diff
        split_results.extend(calculate_stats(mean_results))
        
        # fourth calculation, length of movement in y-loc ts
        ts_num = 1
        fun = ts_stat.derivative_peak_distance(ts_num)
        # calculate the statistic
        if isinstance(conf, tuple):
            mean_results = all_mean_stat_comparison(fun, idx=conf[1], class_dict=conf[2], range_limit=range_subject, test=test)
        else:
            mean_results = all_mean_stat_comparison(fun, config=conf, range_limit=range_subject, test=test)
        # save the pvalue and the mean diff
        split_results.extend(calculate_stats(mean_results))
        
        
        # fifth calculation, length of movement in y-loc ts
        ts_num = 2
        fun = ts_stat.derivative_peak_distance(ts_num)
        # calculate the statistic
        if isinstance(conf, tuple):
            mean_results = all_mean_stat_comparison(fun, idx=conf[1], class_dict=conf[2], range_limit=range_subject, test=test)
        else:
            mean_results = all_mean_stat_comparison(fun, config=conf, range_limit=range_subject, test=test)
        # save the pvalue and the mean diff
        split_results.extend(calculate_stats(mean_results))
        
        
        # add all to the results list
        results.append(split_results)
        
    return results
       
        

if __name__ == "__main__":
    config = "soa"
    idx=1
    dic = {0:0, 6:1}

    ts_num = 1
    start=50
    end=80
    statistics_function = ts_stat.part_trial_mean_creator(ts_num, start, end)
    means = all_mean_stat_comparison(statistics_function=statistics_function, config=config)
    #means = all_mean_stat_comparison(statistics_function=statistics_function, idx=idx, class_dict=dic)

    print(stats.ttest_rel(means[:,0], means[:,1]))