#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 21:41:48 2019

@author: cademay
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, SGD
from sklearn.utils import shuffle
from keras import regularizers 
from keras.callbacks import ModelCheckpoint, EarlyStopping

import category_encoders as ce
oneHotEncoder = ce.OneHotEncoder(cols=[0,1,2])

np.random.seed(133)

ENCODERS_FIT = False
N_CLASSES = 6

AGE_LIMIT = 1991




def binarize(x):
    return 1 if x > 15.0 else 0
        
def bucketize5star(x):
    if x >= 25.0:
        return 5
    elif x >=  20.0:
        return 4
    elif x >=  15.0:
        return 3
    elif x >=  10.0:
        return 2
    elif x >=  5.0:
        return 1
    else:
        return 0
    
def bucketize3(x):
    if x >= 20.0:
        return 2
    elif x >=  10.0:
        return 1
    else:
        return 0
    


def evaluate_predictions(y_pred, y_real, buffer):
    
    correct_count = 0
    n = len(y_pred)
    
    for i in range(n):
        
        pred = y_pred[i]
        real = y_real[i]
        
        if real - buffer <= pred and pred <= real + buffer:
            correct_count += 1
            
    return correct_count / n
    

def get_advanced_stats():
    
    # 2001 -> 2000-2001 season 
    
    advanced_stats_by_year = {}

    for year2 in range(1995,2020):
        
        year1 = year2 - 1
        
        file_name = "data/nba_advanced_stats/"
        file_name += "nba_advanced_" + str(year1) + "_" + str(year2) + ".csv"
        
        statsDF = pd.read_csv(file_name)
        
        advanced_stats_by_year[year2] = statsDF

    return advanced_stats_by_year

def get_max_future_stat(stat_name, cur_player, draft_year, advanced_stats_by_year):
    
    # check season and season +- 1, eg 6, then 5, then 7  if need be, else return nil.
    # must be between 2001 and 2019
    
    
    if draft_year > 2016 or draft_year < AGE_LIMIT:
        return None
    
    player_stats = []
    
    valid_years = advanced_stats_by_year.keys()
    talent_development_buffer = 0 # years
    
    for year in range(draft_year + talent_development_buffer, max(valid_years) + 1):
        
        # if year available
        if year in valid_years:
            
            season_stats = advanced_stats_by_year[year]
            playerList = list(season_stats.Player)
            
            # if player played in that year
            if cur_player in playerList:
                
                player_index = playerList.index(cur_player)
                games_played_in_season = season_stats.at[player_index, "G"]
                
                # if player played substantial number of games that year
                if games_played_in_season > 25:
                    stat_to_return = season_stats.at[player_index, stat_name]
                
                    player_stats.append(stat_to_return)
    
    if len(player_stats) <= 0:
        return None
    
    return max(player_stats)


def get_y(stat_name):
    
    
    y = []
    dropped_player_indeces = []
    
    advanced_stats_by_year = get_advanced_stats()
    
    players = list(X.playerName)
    final_college_seasons = list(X.Season)
    
    for i in range(len(players)): #len(players)
        
        cur_player = players[i]
        draft_year = int(final_college_seasons[i][0:4]) + 1
    
        
        stat_to_predict = get_max_future_stat(stat_name,
                                          cur_player,
                                          draft_year,
                                          advanced_stats_by_year)
        
        if stat_to_predict:
            y.append(stat_to_predict)
        else: 
            dropped_player_indeces.append((i, cur_player, draft_year))

    
    return (y, dropped_player_indeces)



# preproess feature set X 
def preprocess_input(X):

    global ENCODERS_FIT
    global oneHotEncoder
    # drop unwanted features
    
    unwanted_columns = ["playerName", "Season", "spacer1", "transferredSchools", "ORB", "DRB"]  # "\xa0"
    X = X.drop(unwanted_columns, axis=1)

    
    # organize features
    columns = list(X.columns)
    categorical = ["position", "School", "Conf"]
    #diff = lambda l1, l2: [x for x in l1 if x not in l2]
    #numerical = diff(columns, categorical) 
    
    
    # cast numerical data as floats
    for col in columns:
        if col not in categorical:
            X[col] = X[col].astype(float)
    
    
    # impute: fill in missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    X = X.values # convert to ndarray of objects 
    imputer = imputer.fit(X[:, 3:])
    X[:, 3:] = imputer.transform(X[:, 3:])
    
    
    # scale data #
    X_scaler = StandardScaler()
    X_scaler = X_scaler.fit(X[:, 3:])
    X[:, 3:] = X_scaler.transform(X[:, 3:])

    
    
    # encode categorial data #
    
    if not ENCODERS_FIT:
        oneHotEncoder = oneHotEncoder.fit(X)
        X = oneHotEncoder.transform(X)
        ENCODERS_FIT=True
    else:
        X = oneHotEncoder.transform(X)
        

    return X
    
    









# Importing the dataset
college_data = pd.read_csv("data/college_stats/final_CBB_data.csv")

# get X #
X = college_data.copy()

# get y
target_stat = "PER"
y, dropped_player_indeces = get_y(target_stat)


# drop players from X if they didnt qualify for y
for index, player, year in dropped_player_indeces:
    print("dropping ", player)
    X = X.drop([index], axis=0)
 
    
# preprocess X
X_reference = X.copy()
X = preprocess_input(X)
                                          

# scale y

X = np.array(X)
y = np.array(y)

#X, y = shuffle(X,y)




    
if N_CLASSES == 2:
    
    y = [binarize(per) for per in y]
    y = keras.utils.to_categorical(y, 2)

elif N_CLASSES == 3:
    
    y = [bucketize3(per) for per in y]
    y = keras.utils.to_categorical(y, 3)


elif N_CLASSES == 6:
    
    y = [bucketize5star(per) for per in y]
    y = keras.utils.to_categorical(y, 6)









# train-test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_trainR, X_testR, y_trainR, y_testR = train_test_split(X_reference, y, test_size = 0.2, random_state = 0)

input_dim=X_train.shape[1]








def create_custom_model(reg, nHidden, nHiddenUnits, drop):
    
    model = Sequential()
        
    if reg:
        regularizer = regularizers.l2(reg)
    else:
        regularizer = regularizers.l2()
        
        
    model.add(Dense(128, activation = 'relu', input_dim=input_dim,
                    kernel_regularizer=regularizer))
    
    model.add(Dropout(drop))
    
    for i in range(nHidden):
        model.add(Dense(nHiddenUnits, activation = 'relu',
                        kernel_regularizer=regularizer))
        
    model.add(Dense(N_CLASSES, activation = 'softmax',kernel_regularizer=regularizer))
    
    return model


def train_model(model, opt, lr, eps):
    
    if lr:
        Optimizer = Adam(lr=lr) if opt == "Adam" else SGD(lr=lr)
    else:
        Optimizer = Adam() if opt == "Adam" else SGD()
        
    # cb = EarlyStopping(monitor='loss', restore_best_weights=True, patience=200)

    model.compile(optimizer = Optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.fit(x = X_train,
          y = y_train,
          epochs = eps,
          batch_size=64)
    
    return model



def model():
    
    reg = .0001
    lr = None
    opt = "SGD"
    nHidden=2
    nHiddenUnits=128
    epochs = 1200
    dropout=0
    
    # create model architecture
    model = create_custom_model(reg, nHidden, nHiddenUnits, dropout)
    
    # train model
    trained_model = train_model(model, opt, lr, epochs)

    
    return trained_model


model=model()




















model.evaluate(X_test, y_test)


y_pred = np.argmax(model.predict(X_test),axis=1)
y_true = np.argmax(y_test, axis=1)

from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y_true, y_pred)

print (conf)


"""
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

df_cm = pd.DataFrame(conf, index = [i for i in "ABC"],
                  columns = [i for i in "ABC"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
"""


import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels



















def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

if N_CLASSES == 3:
    classes = np.array([ "0 ('Bad')", "1 ('Average')", "2 ('Great')"])

elif N_CLASSES == 6:
    classes = np.array(["0*", "1*","2*", "3*","4*", "5*",])

if N_CLASSES == 2:
    classes = np.array([ "Below Average", "Above Average" ])
    
    
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_true, y_pred, classes=classes,
                      title='3-class Confusion matrix')

# Plot normalized confusion matrix
plot_confusion_matrix(y_true, y_pred, classes=classes, normalize=True,
                      title='Normalized 3-class confusion matrix')
