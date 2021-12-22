import numpy as np
import pandas as pd
from timeseries_data.read_timeseries import read_subject
import timeseries_data.configurations as cfg
import timeseries_data.util.util as util

def data_for_subject_mean_ts():
    data = np.zeros(shape=(280,1000))
    
    data[:140,2] = 0
    data[140:,2] = 1
    
    data[:20, 1] = 0
    data[:20, 3:] = 0
    
    data[20:40, 1] = 1
    data[20:40, 3:] = 1
    
    data[40:60, 1] = 2
    data[40:60, 3:] = 2
    
    data[60:80, 1] = 3
    data[60:80, 3:] = 3
    
    data[80:100, 1] = 4
    data[80:100, 3:] = 4
    
    data[100:120, 1] = 5
    data[100:120, 3:] = 5
    
    data[120:140, 1] = 6
    data[120:140, 3:] = 6
    
    data[140:160, 1] = 0
    data[140:160, 3:] = 7
    
    data[160:180, 1] = 1
    data[160:180, 3:] = 8

    data[180:200, 1] = 2
    data[180:200, 3:] = 9

    data[200:220, 1] = 3
    data[200:220, 3:] = 10

    data[220:240, 1] = 4
    data[220:240, 3:] = 11

    data[240:260, 1] = 5
    data[240:260, 3:] = 12

    data[260:, 1] = 6
    data[260:, 3:] = 13

    return pd.DataFrame(data)


def generate_fake_data():
    for i in range(10):
        data = np.zeros(shape=(280,1323))
    
        data[:140,2] = 0
        data[140:,2] = 1
        
        data[:20, 1] = 0
        data[:20, 120:360] = 1 
        data[:20, 173:203] = 0
        data[:20, 3+240+50:3+240+80] = 0        
        data[:20, 3+1200+40:3+1200+70] = 0
        
        data[20:40, 1] = 1
        data[20:40, 163:203] = -1
        data[20:40, 3+240+50:3+240+80] = -1    
        data[20:40, 3+1200+40:3+1200+70] = -1

        
        data[40:60, 1] = 2
        data[40:60, 153:203] = -2
        data[40:60, 3+240+50:3+240+80] = -2    
        data[40:60, 3+1200+40:3+1200+70] = -2

        
        data[60:80, 1] = 3
        data[60:80, 143:203] = -3
        data[60:80, 3+240+50:3+240+80] = -3    
        data[60:80, 3+1200+40:3+1200+70] = -3

        
        data[80:100, 1] = 4
        data[80:100, 163:203] = -4
        data[80:100, 3+240+50:3+240+80] = -4
        data[80:100, 3+1200+40:3+1200+70] = -4
        
        data[100:120, 1] = 5
        data[100:120, 153:203] = -5
        data[100:120, 3+240+50:3+240+80] = -5    
        data[100:120, 3+1200+40:3+1200+70] = -5
        
        data[120:140, 1] = 6
        data[120:140, 143:203] = -6
        data[120:140, 3+240+50:3+240+80] = -6    
        data[120:140, 3+1200+40:3+1200+70] = -6
        
        data[140:160, 1] = 0
        data[140:160, 173:203] = -7
        data[140:160, 3+240+50:3+240+80] = -7    
        data[140:160, 3+1200+40:3+1200+70] = -7
        
        data[160:180, 1] = 1
        data[160:180, 163:203] = -8
        data[160:180, 3+240+50:3+240+80] = -8    
        data[160:180, 3+1200+40:3+1200+70] = -8
    
        data[180:200, 1] = 2
        data[180:200, 153:203] = -9
        data[180:200, 3+240+50:3+240+80] = -9    
        data[180:200, 3+1200+40:3+1200+70] = -9
        
        data[200:220, 1] = 3
        data[200:220, 143:203] = -10
        data[200:220, 3+240+50:3+240+80] = -10    
        data[200:220, 3+1200+40:3+1200+70] = -10
    
        data[220:240, 1] = 4
        data[220:240, 163:203] = -11
        data[220:240, 3+240+50:3+240+80] = -11    
        data[220:240, 3+1200+40:3+1200+70] = -11
    
        data[240:260, 1] = 5
        data[240:260, 153:203] = -12
        data[240:260, 3+240+50:3+240+80] = -12    
        data[240:260, 3+1200+40:3+1200+70] = -12
    
        data[260:, 1] = 6
        data[260:, 143:203] = -13
        data[260:, 3+240+50:3+240+80] = -13    
        data[260:, 3+1200+40:3+1200+70] = -13
        
        data = pd.DataFrame(data)
        
        data.to_csv(cfg.fake_timeseries_path + "participant" + str (i+1) + ".csv", header=None, index=None)
        

def agency_data_generate():
    data = np.zeros(shape=(240,3))
    
    data[:120,1] = 0
    data[:100,2] = 1
    data[100:120,2] = 0
    
    data[120:140,1] = 1
    data[120:135,2] = 1
    data[135:140,2] = 0
    
    data[140:160,1] = 2
    data[140:150,2] = 1
    data[150:160,2] = 0
    
    data[160:180,1] = 3
    data[160:165,2] = 1
    data[165:180,2] = 0
        
    data[180:200,1] = 4
    data[180:195,2] = 1
    data[195:200,2] = 0
    
    data[200:220,1] = 5
    data[200:210,2] = 1
    data[210:220,2] = 0
    
    data[220:240,1] = 6
    data[220:225,2] = 1
    data[225:240,2] = 0
    
    data = pd.DataFrame(data)
    
    data.to_csv(cfg.timeseries_path + "participant" + str (101) + ".csv", header=None, index=None)
    
    
def special_data_generation():
    data = np.zeros(shape=(140,8))
    
    data[:20, 1] = 0
    data[:15, 2] = 1
    data[:15, 3:] = 0
    data[15:20, 2] = 0
    data[15:20, 3:] = 1
    
    data[20:40, 1] = 1
    data[20:30, 2] = 1
    data[20:30, 3:] = 0
    data[30:40, 2] = 0
    data[30:40, 3:] = 1
    
    data[40:60, 1] = 2
    data[40:45, 2] = 1
    data[40:45, 3:] = 0
    data[45:60, 2] = 0
    data[45:60, 3:] = 1
    
    data[60:80, 1] = 3
    data[60:80, 2] = 0
    data[60:80, 3:] = 1
    
    data[80:100, 1] = 4
    data[80:90, 2] = 1
    data[80:90, 3:] = 0
    data[90:100, 2] = 0
    data[90:100, 3:] = 1
    
    data[100:120, 1] = 5
    data[100:105, 2] = 1
    data[100:105, 3:] = 0
    data[105:120, 2] = 0
    data[105:120, 3:] = 1
    
    data[120:140, 1] = 6
    data[120:140, 2] = 0
    data[120:140, 3:] = 1
    
    data = pd.DataFrame(data)
    
    data.to_csv(cfg.special_feature_path + "participant" + str (100) + ".csv", header=None, index=None)