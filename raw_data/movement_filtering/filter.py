# this function gets subject kinematic data and a filtering function
# it filters the kinematic data and return the number of filtered trials
# and returns movement data, the number of trials that where filtered
def filtering(subject_kinematics, filter_fun):
    # create a list of indices for filtering
    filter_lst = []
    
    for i, (_, trial, *args) in enumerate(subject_kinematics):
        if filter_fun(trial):
            filter_lst.append(i)
    
    # save the number of those numbers 
    num_of_filtered = len(filter_lst)
    
    # delete them from kinematic array
    for i in sorted(filter_lst, reverse=True):
        del subject_kinematics[i]
    
    #print(filter_lst)
    return num_of_filtered
