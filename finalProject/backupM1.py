#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:26:32 2019

@author: cademay
"""

#should binary transferred schools be one-hot encoded or left as binomial?
# imputer check : X[3,4]

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

predictionYears = 6


def get_advanced_stats():
    
    advanced_stats_by_year = {}

    for year2 in range(2001,2020):
        
        year1 = year2 - 1
        
        file_name = "data/nba_advanced_stats/"
        file_name += "nba_advanced_" + str(year1) + "_" + str(year2) + ".csv"
        
        statsDF = pd.read_csv(file_name)
        
        advanced_stats_by_year[year2] = statsDF

    return advanced_stats_by_year

def get_future_stat(stat_name, cur_player, season_to_predict, advanced_stats_by_year):
    
    # check season and season +- 1, eg 6, then 5, then 7  if need be, else return nil.
    # must bebetween 2001 and 2019
    
    years_to_check = [season_to_predict,
                      season_to_predict - 1,
                      season_to_predict + 1,
                      season_to_predict - 2,
                      season_to_predict + 2,
                      season_to_predict - 3,
                      season_to_predict + 3,
                      season_to_predict - 4,
                      season_to_predict + 4
                      ]
    
    valid_years = advanced_stats_by_year.keys()
    
    for year in years_to_check:
        
        if year in valid_years:
            
            season_stats = advanced_stats_by_year[year]
            playerList = list(season_stats.Player)
            
            if cur_player in playerList:
                    
                player_index = playerList.index(cur_player)
                stat_to_return = season_stats.at[player_index, stat_name]
                
                return stat_to_return

    return None


# Importing the dataset
college_data = pd.read_csv("data/college_stats/college_basketball_data_final.csv")


## get X ##

X = college_data.copy()

# get y #
# 2001 -> 2000-2001 season 

y = []

advanced_stats_by_year = get_advanced_stats()

players = list(X.playerName)
final_college_seasons = list(X.Season)
for i in range(len(players)): #len(players)
    
    cur_player = players[i]
    draft_year = int(final_college_seasons[i][0:4]) + 1
    
    season_to_predict = draft_year + predictionYears
    
    stat_name = "PER"
    stat_to_predict = get_future_stat(stat_name,
                                      cur_player,
                                      season_to_predict,
                                      advanced_stats_by_year)
    
    if stat_to_predict:
        y.append(stat_to_predict)
    else:
        print("dropping ", cur_player)
        X = X.drop([i], axis=0)



# drop unwanted features
unwanted_columns = ["playerName", "Season", "\xa0", "transferredSchools"] 
X = X.drop(unwanted_columns, axis=1)

columns = list(X.columns)

diff = lambda l1, l2: [x for x in l1 if x not in l2]
categorical = ["position", "School", "Conf"]
numerical = diff(columns, categorical) 

# cast numerical data as floats
for col in columns:
    if col not in categorical:
        X[col] = X[col].astype(float)


# fill in missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X = X.values # convert to ndarray of objects 
imputer = imputer.fit(X[:, 3:])
X[:, 3:] = imputer.transform(X[:, 3:])





from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


#X = pd.DataFrame(X, columns=columns)


# scale data
X_scaler = StandardScaler()
X_scaler = X_scaler.fit(X[:, 3:])
X[:, 3:] = X_scaler.transform(X[:, 3:])

y_scaler = StandardScaler()
y_scaler = y_scaler.fit(np.array(y).reshape(-1, 1))
y = y_scaler.transform(np.array(y).reshape(-1, 1))


# encode categorial data

labelEncoder_X0 = LabelEncoder()
labelEncoder_X0 = labelEncoder_X0.fit(X[:, 0])
X[:, 0] = labelEncoder_X0.transform(X[:, 0])

labelEncoder_X1 = LabelEncoder()
labelEncoder_X1 = labelEncoder_X1.fit(X[:, 1])
X[:, 1] = labelEncoder_X1.transform(X[:, 1])

labelEncoder_X2 = LabelEncoder()
labelEncoder_X2 = labelEncoder_X2.fit(X[:, 2])
X[:, 2] = labelEncoder_X2.transform(X[:, 2])


oneHotEncoder = OneHotEncoder(categorical_features=[0,1,2])
X = oneHotEncoder.fit_transform(X)







from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



import keras
from keras.models import Sequential
from keras.layers import Dense

def create_model(input_dim):
    
    model = Sequential()
    
    model.add(Dense(64, activation = 'relu', input_dim=input_dim))
    
    model.add(Dense(64, activation = 'relu'))
    
    model.add(Dense(64, activation = 'relu'))
    
    model.add(Dense(1))
    
    return model

model = create_model(X_train.shape[1])


model.compile(loss = "mean_squared_error", optimizer="adam")

model.fit(x = X_train, y = y_train, epochs = 100)


"""
labelEncoder_X0 = LabelEncoder()
labelEncoder_X0 = labelEncoder.fit(X[:, 0])
X[:, 0] = labelEncoder_X0.transform(X[:, 0])

labelEncoder_X1 = LabelEncoder()
labelEncoder_X1 = labelEncoder.fit(X[:, 1])
X[:, 1] = labelEncoder_X1.transform(X[:, 1])

labelEncoder_X2 = LabelEncoder()
labelEncoder_X2 = labelEncoder.fit(X[:, 2])
X[:, 2] = labelEncoder_X2.transform(X[:, 2])


oneHotEncoder = OneHotEncoder(categorical_features=[0,1,2])
X = oneHotEncoder.fit_transform(X)
"""