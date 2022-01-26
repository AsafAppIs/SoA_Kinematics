import numpy as np

from skfeature.function.sparse_learning_based import MCFS
from skfeature.utility import construct_W



def MCFS_filter_creator(num_of_features=10, k=10, n_clusters=2):
    def MCFS_filter(data):
        kwargs_W = {"metric":"cosine","neighbor_mode":"knn","weight_mode":"cosine","k":k,'t':1}
        data = np.array(data)
        W = construct_W.construct_W(data, **kwargs_W)
        score = MCFS.mcfs(data, n_selected_features=num_of_features, W=W, n_clusters=n_clusters)
        idx = MCFS.feature_ranking(score)
    
        idx = idx[:num_of_features]
        return idx
    
    return MCFS_filter



def MCFS_rank_creator(num_of_features=10, k=10, n_clusters=2):
    def MCFS_filter(data):
        kwargs_W = {"metric":"cosine","neighbor_mode":"knn","weight_mode":"cosine","k":k,'t':1}
        data = np.array(data)
        W = construct_W.construct_W(data, **kwargs_W)
        score = MCFS.mcfs(data, n_selected_features=num_of_features, W=W, n_clusters=n_clusters)
        score_max = np.max(score, axis=1)
        return score_max
    
    return MCFS_filter
