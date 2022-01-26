import numpy as np

import feature_engineering.configurations as cfg
from feature_engineering.data_preperation import prepre_data
from feature_engineering.utils.unconfound import soa_unconfound
from feature_engineering.read_features import read_features
from feature_engineering.evaluate_model import evaluate
from sklearn.linear_model import LogisticRegression

# this function return true if the data set isn't passing the threshold
def threshold(labels, threshold):
    unique, counts = np.unique(labels, return_counts=True)
    if len(counts) < 2:
        return True
    return min(counts) < threshold


def analysis(model, test_set=cfg.tests_configurations, thresholds=cfg.class_threshold, feature_mode="clean"):
    results = []
    
    for i in [101]: #cfg.participants_range:
        print(f"analysing subject {i}")
        data = read_features(i, mode=feature_mode)
        subject_results = [i]
        for test in test_set[:12]:
            print(f"test: {test['name']}")
            # filter & label the data accordind to the test 
            X, Y, Z = prepre_data(data, test['filter'], test['labeler'])
            
            # unconfound the data, if necessesry
            if test['unconfound']:
                idx = soa_unconfound(Y, Z)
                X = X[idx]
                Y = Y[idx]
                Z = Z[idx]

            # check whether the test is passing the threshold
            if threshold(Y, thresholds[test['validation']]):
                subject_results.append(-1)
                continue
            
            auc, mat = evaluate(X, Y, model=model, validation_method=test['validation'])
            
            subject_results.append(auc)
            
        results.append(subject_results)
        
    return results



if __name__ == "__main__":
    model =  LogisticRegression()
    res_test1 = analysis(model)