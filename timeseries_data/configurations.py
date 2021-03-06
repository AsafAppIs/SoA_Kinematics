timeseries_path = "C:/Users/User/Documents/asaf/master workspace/Data/timeseries data/" 
fake_timeseries_path = "C:/Users/User/Documents/asaf/master workspace/Data/fake timeseries data/" 

pdf_path = "C:/Users/User/Documents/asaf/master workspace/master_results/mean timeseries/" 
sensitivity_path = "C:/Users/User/Documents/asaf/master workspace/master_results/sensitivity/" 
freq_pdf_path = "C:/Users/User/Documents/asaf/master workspace/master_results/freq mean timeseries/" 
permutation_path = "C:/Users/User/Documents/asaf/master workspace/master_results/permutation distribution/" 
special_feature_path = "C:/Users/User/Documents/asaf/master workspace/Data/special feature data/" 
auc_path = "C:/Users/User/Documents/asaf/master workspace/master_results/auc stats/" 
scatter_path = "C:/Users/User/Documents/asaf/master workspace/master_results/trials_scatter/" 

num_of_original_ts = 6
ts_length = 120
num_of_ts = 22
only_distal = 11
header_size = 3
num_of_coordinates = 3
num_of_participants = 41


distal_idx = [3,4,5,9,10,11,13,17,18,19,21]

class_configurations = ["soa", "manipulation", "manipulation_t", "manipulation_s", 
                        ("manipulation_t1", 1, {0:0, 1:1}), ("manipulation_t2", 1, {0:0, 2:1}),("manipulation_t3", 1, {0:0, 3:1}),
                        ("manipulation_s1", 1, {0:0, 4:1}), ("manipulation_s2", 1, {0:0, 5:1}),("manipulation_s3", 1, {0:0, 6:1}),
                        ("unconsious manipulation t1", [1,2], {(0,1):0, (1,1):1}), ("unconsious manipulation t2", [1,2], {(0,1):0, (2,1):1}),
                        ("unconsious manipulation s1", [1,2], {(0,1):0, (4,1):1}), ("unconsious manipulation s2", [1,2], {(0,1):0, (5,1):1}),
                        ("internal unconsious manipulation t1", [1,2], {(1,0):0, (1,1):1}), ("internal unconsious manipulation t2", [1,2], {(2,0):0, (2,1):1}),
                        ("internal unconsious manipulation s1", [1,2], {(4,0):0, (4,1):1}), ("internal unconsious manipulation s2", [1,2], {(5,0):0, (5,1):1}),
                        "random"]

sensitivity_configurations = [('general', 2, {0:1, 1:1}), ('temporal', 1, {0:1, 1:1, 2:1, 3:1}), ('spatial', 1, {0:1, 4:1, 5:1, 6:1}),
                              ('temporal1', 1, {0:1, 1:1}), ('temporal2', 1, {0:1, 2:1}), ('temporal3', 1, {0:1, 3:1}),
                              ('spatial1', 1, {0:1, 4:1}), ('spatial2', 1, {0:1, 5:1}), ('spatial3', 1, {0:1, 6:1}),
                              ("unconsious temporal1", [1,2], {(0,1):1, (1,1):1}), ("unconsious temporal2", [1,2], {(0,1):1, (2,1):1}),
                              ("unconsious spatial1", [1,2], {(0,1):1, (4,1):1}), ("unconsious spatial2", [1,2], {(0,1):1, (5,1):1}),
                              ]

sensitivity_names = [x[0] for x in sensitivity_configurations]
sensitivity_names_dp = [x + "_dp" for x in sensitivity_names]
sensitivity_names_bias = [x + "_bias" for x in sensitivity_names]
sensitivity_names_cols = [None]*(len(sensitivity_names_bias)+len(sensitivity_names_dp))
sensitivity_names_cols[::2] = sensitivity_names_dp
sensitivity_names_cols[1::2] = sensitivity_names_bias

#class_configurations = ["manipulation_t"]

class_configurations_names = [x if isinstance(x, str) else x[0] for x in class_configurations ]

full_names = ['M_X_L', 'M_Y_L', 'M_Z_L',
         'D_X_L', 'D_Y_L', 'D_Z_L',
         'M_X_V', 'M_Y_V', 'M_Z_V',
         'D_X_V', 'D_Y_V', 'D_Z_V',
         'M_T_V', 'D_T_V',
         'M_X_A', 'M_Y_A', 'M_Z_A',
         'D_X_A', 'D_Y_A', 'D_Z_A',
         'M_T_A', 'D_T_A']


distal_names = ['D_X_L', 'D_Y_L', 'D_Z_L',
         'D_X_V', 'D_Y_V', 'D_Z_V', 'D_T_V',
         'D_X_A', 'D_Y_A', 'D_Z_A', 'D_T_A']


SAMPLE_RATE = 60  # Hertz
DURATION = 2  # Seconds
