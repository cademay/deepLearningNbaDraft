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

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

np.random.seed(42)



AGE_LIMIT = 1991

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


def get_y():
    
    
    y = []
    dropped_player_indeces = []
    
    advanced_stats_by_year = get_advanced_stats()
    
    players = list(X.playerName)
    final_college_seasons = list(X.Season)
    
    for i in range(len(players)): #len(players)
        
        cur_player = players[i]
        draft_year = int(final_college_seasons[i][0:4]) + 1
    
        
        stat_name = "WS"
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
def preprocess_input(X, le0, le1, le2, oHot, supplemental_label_data=pd.DataFrame()):
    
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
    
    fit_data0 = X[:, 0]
    fit_data1 = X[:, 1]
    fit_data2 = X[:, 2]
    print(fit_data0)
    print(fit_data0.shape)
        
    if len(supplemental_label_data) > 0:
        
        fit_data0 = list(fit_data0)
        fit_data0.extend(list(supplemental_label_data.School))
        
        print("yale:",fit_data0.index('Yale'))
        fit_data1 = list(fit_data1)
        fit_data1.extend(list(supplemental_label_data.Conf))
        
        fit_data2 = list(fit_data2)
        fit_data2.extend(list(supplemental_label_data.position))

        
    if not le0:
        # label encode
        labelEncoder_X0 = LabelEncoder()
        labelEncoder_X0 = labelEncoder_X0.fit(fit_data0)
        X[:, 0] = labelEncoder_X0.transform(X[:, 0])
        
        labelEncoder_X1 = LabelEncoder()
        labelEncoder_X1 = labelEncoder_X1.fit(fit_data1)
        X[:, 1] = labelEncoder_X1.transform(X[:, 1])
        
        labelEncoder_X2 = LabelEncoder()
        labelEncoder_X2 = labelEncoder_X2.fit(fit_data2)
        X[:, 2] = labelEncoder_X2.transform(X[:, 2])
        
        # one hot encode
        oneHotEncoder = OneHotEncoder(categorical_features=[0,1,2])
        oneHotEncoder = oneHotEncoder.fit(X)
        X = oneHotEncoder.transform(X)
    else:
        # label encoder with prev encoder
        X[:, 0] = le0.transform(X[:, 0])
        X[:, 1] = le1.transform(X[:, 1])
        X[:, 2] = le2.transform(X[:, 2])
        
        # one hot encode with prev encoder
        X = oHot.transform(X)
        
        

    return (X, labelEncoder_X0, labelEncoder_X1, labelEncoder_X2, oneHotEncoder)
    
    





# Importing the dataset
college_data = pd.read_csv("data/college_stats/final_CBB_data.csv")
#college_data = pd.read_csv("data/leftovers/college_basketball_data_final.csv")

prospects_2019_data = pd.read_csv(
        "data/college_stats/college_basketball_data_2019_draft_prospects_final.csv"
        )

# get X #
X = college_data.copy()


# get y #
y, dropped_player_indeces = get_y()


# drop players from X if they didnt qualify for y
for index, player, year in dropped_player_indeces:
    
    print("dropping ", player)
    X = X.drop([index], axis=0)
 
    
# preprocess X
supp = prospects_2019_data
X_reference = X.copy()
X, le0, le1, le2, oHot = preprocess_input(X,
                                          None, None, None, None, supp)
                                          
            

# scale y
y_scaler = StandardScaler()
y_scaler = y_scaler.fit(np.array(y).reshape(-1, 1))
y = y_scaler.transform(np.array(y).reshape(-1, 1))





# train-test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_trainR, X_testR, y_trainR, y_testR = train_test_split(X_reference, y, test_size = 0.2, random_state = 0)







def create_model(input_dim):
    
    model = Sequential()
        
    model.add(Dense(128, activation = 'relu', input_dim=input_dim))
    model.add(Dropout(.2))
    
    for i in range(3):
        model.add(Dense(128, activation = 'relu'))
        model.add(Dropout(.2))

    
    model.add(Dense(1))
    
    return model



# train model
model = create_model(X_train.shape[1])
model.compile(loss = "mean_squared_error", optimizer="adam")
model.fit(x = X_train, y = y_train, epochs = 200)



model.evaluate(X_test, y_test)

y_pred = y_scaler.inverse_transform(model.predict(X_test))
y_real = y_scaler.inverse_transform(y_test)


# calculate percentage of winshares within += buffer of real ###
def evaluate_predictions(y_pred, y_real, buffer):
    
    correct_count = 0
    n = len(y_pred)
    
    for i in range(n):
        
        pred = y_pred[i]
        real = y_real[i]
        
        if real - buffer <= pred and pred <= real + buffer:
            correct_count += 1
            
    return correct_count / n
    

buffer = 2.5
accuracy = evaluate_predictions(y_pred, y_real, buffer)
print(accuracy)



prospects_2019_data = pd.read_csv(
        "data/college_stats/college_basketball_data_2019_draft_prospects_final.csv"
        )
X_2019_reference = prospects_2019_data.copy()
X_2019 = preprocess_input(prospects_2019_data, le0, le1, le2, oHot)


pred_WS_2019 = model.predict(X_2019)






#https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
from matplotlib.pyplot import scatter
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

test_names = list(X_testR.playerName)

predDF = pd.DataFrame()
predDF["names"] = test_names
predDF["pred max WS"] = y_real

predDF = predDF.sort_values(by="pred max WS", ascending=False)



snDF = pd.DataFrame()
snDF["names"] = test_names
snDF["real max WS"] = y_real



import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


predDF = predDF.values
vals = predDF[0:8, 1]
pls = predDF[0:8, 0]

objects = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
y_pos = np.arange(len(pls))
performance = [10,8,6,4,2,1]

plt.barh(y_pos, vals, align='center', alpha=0.5)
plt.yticks(y_pos, pls)





for i in range(len(test_names)):
    print(test_names[i])
    print(y_real[i])

good_bool = (y_real > 13.0)
good_inds = [x for x, y in enumerate(good_bool) if y == True]


good_names = [test_names[i] for i in good_inds]
good_pred = [y_pred[i][0] for i in good_inds]
good_real = [y_real[i][0] for i in good_inds]


# how to add y=x line to this?
plt.scatter(good_real, good_pred, alpha=0.2, cmap='viridis')
plt.xlabel("real WS")
plt.ylabel("predicted WS")


fig, ax = plt.subplots()
ax.scatter(good_real, good_pred, alpha=0.2, cmap='viridis')

ind = int(.8*len(list(X_reference.playerName)))-1

for i, txt in enumerate(good_names):
    
    ax.annotate(txt, (good_real[i], good_pred[i]))

ax.set_xlabel("real WS")
ax.set_ylabel("predicted WS")




import seaborn as sns
sns.set(style="whitegrid")

graphData = pd.DataFrame()

graphData["player"] = good_names
graphData["real"] = good_real
graphData["pred"] = good_pred



# Draw a nested barplot to show survival for class and sex
g = sns.catplot(x="real", y="pred", hue="player", data=graphData,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("Win Shares")


names = []
vals = []
WS = []


def getName(player):
    
    player = player.replace(".", "") # eg J.J. Redick
    player = player.replace("'", "") # Shaquille O'Neal*\onealsh01"
    player = player.replace("*", "") # Shaquille O'Neal*\onealsh01"
    
    
    spaceIndex = player.index(" ")
    slashIndex = player.index("\\")
    
    player = player.lower()
        
    firstName = player[0 : spaceIndex]
    lastName = player[spaceIndex + 1 : slashIndex ]
    
    return (firstName, lastName)

def initials(name):
    f, l = getName(name)
    
    return f[0].upper() + l[0].upper() 
    

for i in range(len(good_names)):
    
    names.append(initials(good_names[i]))
    vals.append(good_real[i])
    WS.append("real")
    
for i in range(len(good_names)):
    
    names.append(initials(good_names[i]))
    vals.append(good_pred[i])
    WS.append("predicted")

graphData = pd.DataFrame()
graphData["player"] = names
graphData["value"] = vals
graphData["WS"] = WS


import seaborn as sns
sns.set(style="whitegrid")
tips = sns.load_dataset("tips")
#ax = sns.barplot(x="day", y="total_bill", data=tips)

#ax = sns.barplot(x="day", y="total_bill", hue="sex", data=tips)


ax = sns.barplot(x="player", y="value", hue="WS", data=graphData)
print(good_names)





import matplotlib.pyplot as plt


x = y_real
y = y_pred
names = test_names
#c = np.random.randint(1,5,size=len(y_real))

norm = plt.Normalize(1,4)
cmap = plt.cm.RdYlGn

fig,ax = plt.subplots()
sc = plt.scatter(x,y, s=100, cmap=cmap, norm=norm, alpha=.1)

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))

annot.set_visible(False)

def update_annot(ind):

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
                           " ".join([names[n] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()








import plotly.plotly as py
import plotly.graph_objs as go


random_x = y_real
random_y = y_pred

# Create a trace
trace = go.Scatter(
    x = random_x,
    y = random_y,
    mode = 'markers'
)

data = [trace]

# Plot and embed in ipython notebook!
py.iplot(data, filename='basic-scatter')










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


