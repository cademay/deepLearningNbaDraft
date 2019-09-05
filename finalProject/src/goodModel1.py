
#should binary transferred schools be one-hot encoded or left as binomial?
# imputer check : X[3,4]


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

AGE_LIMIT = 1991
N_CLASSES=3

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
    



def getName(player):
    
    player = player.replace(".", "") # eg J.J. Redick
    player = player.replace("'", "") # Shaquille O'Neal*\onealsh01"
    player = player.replace("*", "") # Shaquille O'Neal*\onealsh01"
    
    
    spaceIndex = player.index(" ")
    slashIndex = player.index("\\")
    
 
        
    firstName = player[0 : spaceIndex]
    lastName = player[spaceIndex + 1 : slashIndex ]
    
    return (firstName, lastName)


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

def get_max_future_stat(stat_name, cur_player, draft_year, advanced_stats_by_year, ind):
    
    # check season and season +- 1, eg 6, then 5, then 7  if need be, else return nil.
    # must be between 2001 and 2019
    
    
    if draft_year > 2016 or draft_year < AGE_LIMIT:
        return None
    
    if ind%2==0 and dups[ind]:
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
    indeces = list(X.index)
    final_college_seasons = list(X.Season)
    print(len(players), "========")
    for i in range(len(players)): #len(players)
        
        cur_player = players[i]
        draft_year = int(final_college_seasons[i][0:4]) + 1
    
        
        stat_to_predict = get_max_future_stat(stat_name,
                                          cur_player,
                                          draft_year,
                                          advanced_stats_by_year,
                                          indeces[i])
        
        if stat_to_predict:
            y.append(stat_to_predict)
        else: 
            dropped_player_indeces.append((indeces[i], cur_player, draft_year))

    
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
X=shuffle(X)
#X.drop_duplicates("playerName", keep="first", inplace=True)
dups=X.duplicated("playerName", keep="first")





# get y
target_stat = "VORP"
y, dropped_player_indeces = get_y(target_stat)


# drop players from X if they didnt qualify for y
for index, player, year in dropped_player_indeces:
    print("dropping ", player)
    X = X.drop([index], axis=0)
 
    
    
# preprocess X
X_reference = X.copy()
X = preprocess_input(X)
                                          

# scale y
y_scaler = StandardScaler()
y_scaler = y_scaler.fit(np.array(y).reshape(-1, 1))
y = y_scaler.transform(np.array(y).reshape(-1, 1))



X = np.array(X)
y = np.array(y)

X, y = shuffle(X,y)


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
        
        
    model.add(Dense(64, activation = 'relu', input_dim=input_dim,
                    kernel_regularizer=regularizer))
    
    model.add(Dropout(drop))
    
    for i in range(nHidden):
        model.add(Dense(nHiddenUnits, activation = 'relu',
                        kernel_regularizer=regularizer))
        
    model.add(Dense(1))
    
    return model


def train_model(model, opt, lr, eps):
    
    if lr:
        Optimizer = Adam(lr=lr) if opt == "Adam" else SGD(lr=lr)
    else:
        Optimizer = Adam() if opt == "Adam" else SGD()
    
    model.compile(loss = "mean_squared_error", optimizer=Optimizer)
    model.fit(x = X_train,
          y = y_train,
          epochs = eps)
    
    return model



def model():
    """
    reg = .0001
    lr = None
    opt = "Adam"
    nHidden=2
    nHiddenUnits=128
    epochs = 1500
    dropout=0
    """
    
    reg = .0001
    lr = .001
    opt = "Adam"
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











print(model.evaluate(X_test,y_test))




y_pred = y_scaler.inverse_transform(model.predict(X_test)).reshape(len(X_test))
y_real = y_scaler.inverse_transform(y_test).reshape(len(X_test))


buffer = 2.5
accuracy = evaluate_predictions(y_pred, y_real, buffer)
print(accuracy)























import seaborn as sns
sns.set()
import matplotlib.pyplot as plt



pos = X_testR.position
# plot
ax = sns.scatterplot(x=y_real, y=y_pred, alpha=.72) 
ax.set(xlabel="Real VORP" , ylabel="Predicted VORP")
ax.set(title="Predicted NBA VORP vs. Real NBA VORP ")



















####################




imput_dim=X_train.shape[1]

reg = 1e-5


def create_model(input_dim):
    
    model = Sequential()
        
    model.add(Dense(128, activation = 'relu', input_dim=input_dim,
                    kernel_regularizer=regularizers.l2(reg)))
    #model.add(Dropout(.1))
    
    for i in range(4):
        model.add(Dense(64, activation = 'relu',
                        kernel_regularizer=regularizers.l2(reg)))
        #model.add(Dropout(.1))

    model.add(Dense(1))
    
    return model



# train model
model = create_model(X_train.shape[1])
model.compile(loss = "mean_squared_error", optimizer=SGD())
model.fit(x = X_train, y = y_train, epochs = 600)


##########






