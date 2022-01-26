import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from skfeature.function.sparse_learning_based import MCFS
from skfeature.utility import construct_W

import feature_engineering.configurations as cfg
from feature_engineering.read_features import read_features
from feature_engineering.MCFS import MCFS_filter_creator


pd.options.mode.chained_assignment = None

def random_filter_creator(num_of_features=50):
    
    def stupid_filter(data):
        idx = np.arange(0, data.shape[1],1)
        np.random.shuffle(idx)
        
        idx = idx[:num_of_features] 
        return idx
    
    return stupid_filter 





def toy_labeler(data, label):
    data = data.iloc[:, 4:]
    data.loc[:, 'label'] = label
    return data

def evaluate(data, filter_fun, calculate=False):   
    X = data.loc[:, data.columns != 'label']
    Y = data['label']
    
    
    
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=.2, random_state=42)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    
    idx = filter_fun(x_train)
    x_train = x_train[:, idx]
    x_test = x_test[:, idx]
    
    lrm = LogisticRegression()
    lrm.fit(x_train, y_train)
    y_hat = lrm.predict(x_test)
    
    if calculate:
        return accuracy_score(y_test, y_hat)
    
    print(classification_report(y_test, y_hat))
    print(confusion_matrix(y_test, y_hat))

def prepare_data(idx_lst):
    data = [read_features(x) for x in idx_lst]
    data = [toy_labeler(x, np.random.choice([0,1])) for x in data]

    
    data = pd.concat(data, axis=0)
    
    return data
    
    
def random_filtering_test():
    num_of_tests = 100
    num_of_features = np.arange(10,51,2)
    results = np.zeros((len(num_of_features), num_of_tests))
    data = prepare_data(1,4)
    for i, num in enumerate(num_of_features):
        print(f"testing classifier performance with {num} random features")
        fun = random_filter_creator(num)
        for j in range(num_of_tests):
           results[i,j] =  evaluate(data, fun, calculate=True)
           
    return results



def MCFS_filtering_test():
    num_of_features = np.arange(10,51,2)
    results = np.zeros((len(num_of_features)))
    data = prepare_data(4,1)
    for i, num in enumerate(num_of_features):
        print(f"testing classifier performance with {num} random features")
        fun = MCFS_filter_creator(num)
        results[i] =  evaluate(data, fun, calculate=True)
        
    return results
                       


def toy_hyperparameter_tuning():
    k_list = np.arange(3,30,1)
    results = np.zeros((len(k_list), 2))
    data = prepare_data(3,2)
    for i, k in enumerate(k_list):
        print(f"testing classifier performance with k={k}")
        fun = MCFS_filter_creator(10, k)    
        results[i] =  evaluate(data, fun, calculate=True)
        
    return results
                       
data = prepare_data(range(1,2))
fun = MCFS_filter_creator(10, 10)         
evaluate(data, fun, calculate=False)
    
