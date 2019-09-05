#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 20:28:25 2019

@author: cademay
"""

import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

nba_data_2017_18 = pd.read_csv("nbaAdvanced2017-18.csv")

print(nba_data_2017_18.columns)

per = np.array(nba_data_2017_18.PER)


tt = [5,10,5,0,1,2,3,4,5,6,7]
t = tt > 4
