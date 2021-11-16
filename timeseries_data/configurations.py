timeseries_path = "C:/Users/User/Documents/asaf/master workspace/Data/timeseries data/" 
pdf_path = "C:/Users/User/Documents/asaf/master workspace/master_results/mean timeseries/" 

num_of_original_ts = 6
ts_length = 120
num_of_ts = 22
only_distal = 11
header_size = 3
num_of_coordinates = 3
num_of_participants = 41


distal_idx = [3,4,5,9,10,11,13,17,18,19,21]

class_configurations = ["soa", "manipulation", "manipulation_t", "manipulation_s", "random",
                        ("manipulation_min_t", 1, {0:0, 1:1}), ("manipulation_min_s", 1, {0:0, 4:1}),
                        ("manipulation_max_t", 1, {0:0, 3:1}), ("manipulation_max_s", 1, {0:0, 6:1}),]

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
