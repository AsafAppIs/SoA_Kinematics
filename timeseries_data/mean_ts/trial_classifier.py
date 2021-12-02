from random import randint, seed

# this function create a 'trial classifier' function that classifying trials into classes 
# this function either gets a string that represent premade trial classification 
# or an index and dictionary to create custom classification 
# the function return '0' for the first class. '1' for the second and '2' for trials that will be excluded
def trial_classfier_creator(config=False, idx=0, class_dict=False):
    # check whether both config and dictionary are defined, if so, assert an error
    
    assert not (config and class_dict), "either configuration string or custom dictionary could be defined"
    
    # define the dict and the idx according to the configuration string, if defined
    if config == "soa":
        idx = 2
        class_dict = {0:1, 1:0}
    elif config == "manipulation":
        idx = 1
        class_dict = {0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1}
    elif config == "manipulation_t":
        idx = 1
        class_dict = {0:0, 1:1, 2:1, 3:1}
    elif config == "manipulation_s":
        idx = 1
        class_dict = {0:0, 4:1, 5:1, 6:1}
    elif config == "random":
        idx = 0
      
    
    # seed for reproductability
    #seed(10)
    
    # define the classifier function
    def classifer(trial_header):
        if not idx:
            return randint(0, 1)
        if isinstance(idx, int):
            key = trial_header[idx]
        else:
            key = tuple(trial_header[idx])
        return class_dict.get(key, 2)
    
    return classifer