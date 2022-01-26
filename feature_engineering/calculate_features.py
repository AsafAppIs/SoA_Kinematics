import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

import feature_engineering.configurations as cfg
from feature_engineering.read_features import read_features
from feature_engineering.MCFS import MCFS_filter_creator, MCFS_rank_creator

from skfeature.function.sparse_learning_based import MCFS
from skfeature.utility import construct_W

def data_preparation(participant_num):
    data = read_features(participant_num)
    X = data.iloc[:, cfg.header_size:]
    
    sc = StandardScaler()
    features = sc.fit_transform(X)
    
    return features

def load_data():
    data = [read_features(x) for x in range(1, 42)]
    data = pd.concat(data, axis=0)
    
    X = data.iloc[:, cfg.header_size:]
    
    sc = StandardScaler()
    features = sc.fit_transform(X)
        
    return features


def top_score(results, scores, top=50):
    score_max = [(i,x) for i,x in enumerate(scores)]
    sorted_score_max = sorted(score_max, key=lambda x: x[1], reverse=True)
    for i in range(top):
        results[sorted_score_max[i][0]] += 1
        
    return results


def add_scores(results, scores):
    return np.max([results, scores], axis=0)


def calculate_subject_features_scores(participant_num, k=10, n_clusters=7):
    features = data_preparation(participant_num)
    fun = MCFS_rank_creator(num_of_features=100, k=k, n_clusters=n_clusters)
    
    score = fun(features)
    
    return score


@ignore_warnings(category=ConvergenceWarning)
def calculate_features_scores(k=10, n_clusters=7):
    results = np.zeros(shape=(cfg.num_of_features))
    
    for i in range(cfg.num_of_participants):
        print(f"subject {i+1}")
        score = calculate_subject_features_scores(i+1, k=k, n_clusters=n_clusters)
        results = top_score(results, score)
        
    
    results_max = [(i,x) for i,x in enumerate(results)]
    sorted_results_max = sorted(results_max, key=lambda x: x[1], reverse=True)
    
    return sorted_results_max



@ignore_warnings(category=ConvergenceWarning)
def calculate_features_scores_combained(k=10, n_clusters=7):
    features = load_data()
    print("finish data loading")
    fun = MCFS_rank_creator(num_of_features=100, k=k, n_clusters=n_clusters)
    
    score = fun(features)
    
    results_max = [(i,x) for i,x in enumerate(score)]
    sorted_results_max = sorted(results_max, key=lambda x: x[1], reverse=True)
    
    return sorted_results_max

    
    
    
results = calculate_features_scores_combained()
