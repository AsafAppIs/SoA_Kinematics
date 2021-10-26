# -*- coding: utf-8 -*-
import pandas as pd
from copy import deepcopy
from random import randint
LINE_PATH = "C:/Users/User/Documents/asaf\master workspace/Data/raw data/line.csv" 
LINE = pd.read_csv(LINE_PATH, header=None)

def return_line():
    return deepcopy(LINE)

def return_kinematic_line():
    line = deepcopy(LINE)
    line.iloc[0,5] = 'ShowingHands'
    line.iloc[0,6] = 'Execution'
    return line
    
def wraping_trial():
    df = return_line()
    for i in range(49):
        df = df.append(return_line())
    return df

def movememt_trial(step_size, length=121):
    df = return_kinematic_line()
    for i in range(length-1):
        line = df.iloc[-1]
        movemnts_axis = i % 3
        line[30+movemnts_axis] += step_size
        line[33+movemnts_axis] += step_size * 2
        df = df.append(line)
    return df

def flip_trial(trial):
    idx = randint(1, len(trial))
    trial.iloc[idx,8] = False
    return trial

def create_trial(step_size, flip, num):
    df = wraping_trial()
    movement = movememt_trial(step_size)
    if flip:
        movement = flip_trial(movement)
    df = df.append(movement)
    df = df.append(wraping_trial())
    df.iloc[:,2] = num
    return df


def create_subject(flipped, no_movement, almost_no_movement, 
               exeggarted_movement, step_size, length=100):
    df = create_trial(step_size, False, 1)
    for i in range(length-1):
        step = step_size
        is_flip = False
        if i+2 in flipped:
            is_flip = True
        if i+2 in no_movement:
            step = 0
        if i+2 in almost_no_movement:
            step = step/10
        if i+2 in exeggarted_movement:
            step = 1
        trial = create_trial(step, is_flip, i+2)
        df = df.append(trial)
    return df
        
            