timeseries_path = "C:/Users/User/Documents/asaf/master workspace/Data/timeseries data/" 
full_feature_path = "C:/Users/User/Documents/asaf/master workspace/Data/full feature data/" 
full_clean_feature_path = "C:/Users/User/Documents/asaf/master workspace/Data/full clean feature data/" 
filtered_feature_path = "C:/Users/User/Documents/asaf/master workspace/Data/filtered feature data/" 
clean_results_path = "C:/Users/User/Documents/asaf/master workspace/master_results/classification results/Logistic Regression/clean feature mode/" 


header_size = 3
num_of_features = 8657
num_of_participants = 41
cluster_size = [2,4,6,8]
k_list = [5,7,10,15,20,25]

k_validation = 10

random_seed = 42
np_seed = 42

class_threshold = {'cv': 18, 'lto': 12}

participants_range = range(1, num_of_participants + 1)


tests_configurations = [{'name': 'all manipulation', 'filter' :(1, [0,1,2,3]), 'labeler': (1, {0:0, 1:1, 2:1, 3:1}), 'validation':'cv', 'unconfound' : False},
                        {'name': 'all agency', 'filter' :(1, [0,1,2,3]), 'labeler': (2, {0:0, 1:1}), 'validation':'cv', 'unconfound' : False},
                        {'name': 'all agency unconfounded', 'filter' :(1, [0,1,2,3]), 'labeler': (2, {0:0, 1:1}), 'validation':'cv', 'unconfound' : True},
                        {'name': '100ms manipulation', 'filter' :(1, [0,1]), 'labeler': (1, {0:0, 1:1}), 'validation':'cv', 'unconfound' : False},
                        {'name': '100ms agency', 'filter' :(1, [0,1]), 'labeler': (2, {0:0, 1:1}), 'validation':'cv', 'unconfound' : False},
                        {'name': '100ms agency unconfounded', 'filter' :(1, [0,1]), 'labeler': (2, {0:0, 1:1}), 'validation':'cv', 'unconfound' : True},
                        {'name': '200ms manipulation', 'filter' :(1, [0,2]), 'labeler': (1, {0:0, 2:1}), 'validation':'cv', 'unconfound' : False},
                        {'name': '200ms agency', 'filter' :(1, [0,2]), 'labeler': (2, {0:0, 1:1}), 'validation':'cv', 'unconfound' : False},
                        {'name': '200ms agency unconfounded', 'filter' :(1, [0,2]), 'labeler': (2, {0:0, 1:1}), 'validation':'cv', 'unconfound' : True},
                        {'name': '300ms manipulation', 'filter' :(1, [0,3]), 'labeler': (1, {0:0, 3:1}), 'validation':'cv', 'unconfound' : False},
                        {'name': '300ms agency', 'filter' :(1, [0,3]), 'labeler': (2, {0:0, 1:1}), 'validation':'cv', 'unconfound' : False},
                        {'name': '300ms agency unconfounded', 'filter' :(1, [0,3]), 'labeler': (2, {0:0, 1:1}), 'validation':'cv', 'unconfound' : True},
                        {'name': '0ms inside agency', 'filter' :(1, [0]), 'labeler': (2, {0:0, 1:1}), 'validation': 'lto', 'unconfound' : False},
                        {'name': '100ms inside agency', 'filter' :(1, [1]), 'labeler': (2, {0:0, 1:1}), 'validation': 'lto','unconfound' : False},
                        {'name': '200ms inside agency', 'filter' :(1, [2]), 'labeler': (2, {0:0, 1:1}), 'validation':'lto', 'unconfound' : False},
                        {'name': '300ms inside agency', 'filter' :(1, [3]), 'labeler': (2, {0:0, 1:1}), 'validation': 'lto', 'unconfound' : False},
                        {'name': 'unconcious manipulation', 'filter' :([1,2], [(0,1), (1,1)]), 'labeler': (1, {0:0, 1:1}), 'validation':'lto' , 'unconfound': False},]

test_names = ['id'] + [x['name'] for x in tests_configurations]