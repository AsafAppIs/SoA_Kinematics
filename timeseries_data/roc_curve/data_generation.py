import numpy as np
import pandas as pd

import timeseries_data.configurations as cfg

line11 = [0,0,1,-1,0,0]
line10 = [0,0,1,0,0,0]
line01 = [0,0,0,-1,0,0]
line00 = [0,0,0,0,0,0]

data11 = [line11 for i in range(50)]
data10 = [line10 for i in range(50)]
data01 = [line01 for i in range(50)]
data00 = [line00 for i in range(50)]

fake1 = pd.DataFrame(data11+data00)
fake2 = pd.DataFrame(data11+data00+data10+data01)

fake1.to_csv(cfg.special_feature_path + "fake1.csv", header=None, index=None)
fake2.to_csv(cfg.special_feature_path + "fake2.csv", header=None, index=None)