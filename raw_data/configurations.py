# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 11:25:17 2021

@author: User
"""

data_path = "C:/Users/User/Documents/asaf\master workspace/Data/raw data" 
first_data = "First experiment"
second_data = "Second experiment"

first_participants = [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27]
second_participants = [1,2,4,5,6,7,8,9,10,11,12,13,14,15,17,19,20,21,22,23,24,25,26,28,29,31,32,33,34,35,36,38,39,40,41,42,43]

first_participants = [(1,x) for x in first_participants]
second_participants  = [(2,x) for x in second_participants]
all_participants = first_participants + second_participants 


#all_participants = [(1,98),(1,99)]
#all_participants = all_participants[:6]

num_of_std_from_mean = 3