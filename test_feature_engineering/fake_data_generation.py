import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, LeavePOut
from sklearn.preprocessing import StandardScaler

TOTAL = 1500
RATIO = [.8, .6, .4, .2]
FEATURES = 50
SCALING = 0.1

def participant_generation():
    data = []
    for manipulation in [0,1,2,3]:
        for agency in [0,1]:
            length = round(TOTAL * abs((1-agency) - RATIO[manipulation]))  
            print(length)
            header = np.array([[0,manipulation,agency] for i in range(length)])
            features_agency = np.random.normal(loc=agency*SCALING,  size=(length, FEATURES))
            features_manipulation = np.random.normal(loc=manipulation*SCALING, size=(length, FEATURES))
            condition = np.concatenate((header, features_agency, features_manipulation), axis=1)
            data.append(condition)
            
    data = np.concatenate(data, axis=0)
    
    return data
            


def generate_data(mean=1):
    class_size = 1500
    features = 50
    
    class0 = np.random.normal(loc=0, scale=1, size=(class_size, features))
    class1 = np.random.normal(loc=mean, scale=1, size=(class_size, features))
    
    label_0 = np.zeros(class_size)
    label_1 = np.ones(class_size)
    
    X = np.concatenate((class0, class1), axis=0)
    
    Y = np.concatenate((label_0, label_1))
    
    return X, Y
    


def evaluate(X, Y):
    results = []
    kf = StratifiedKFold(n_splits=10)    
    # iterate over the folds
    for train_index, test_index in kf.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        # standartize data
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
        
        # fit model        
        model = LogisticRegression()
        model.fit(x_train, y_train)
        
        y_hat = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_hat)
        
        results.append(accuracy)
        
    return np.mean(results)

data = participant_generation()