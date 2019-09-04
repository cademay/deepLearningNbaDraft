
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

import category_encoders as ce
oneHotEncoder = ce.OneHotEncoder(cols=[0,1,2])

np.random.seed(42)

ENCODERS_FIT = False

AGE_LIMIT = 2008



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

def get_max_future_stat(stat_name, cur_player, draft_year, advanced_stats_by_year, position, position_filter=None):
    
    # check season and season +- 1, eg 6, then 5, then 7  if need be, else return nil.
    # must be between 2001 and 2019
    
    
    if draft_year > 2016 or draft_year < AGE_LIMIT:
        return None
    
    if position_filter and position != position_filter:
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


def get_y(data, stat_name, position_filter=None):
    
    
    y = []
    dropped_player_indeces = []
    
    advanced_stats_by_year = get_advanced_stats()
    
    players = list(data.playerName)
    final_college_seasons = list(data.Season)
    positions = list(data.position)
    
    for i in range(len(players)): #len(players)
        
        cur_player = players[i]
        draft_year = int(final_college_seasons[i][0:4]) + 1
        position = positions[i]
    
        
        stat_to_predict = get_max_future_stat(stat_name,
                                          cur_player,
                                          draft_year,
                                          advanced_stats_by_year,
                                          position,
                                          position_filter)
        
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
    
    
def get_nba_player_names(year_start, year_end):

    player_list = []
    
    for yr1 in range(year_start, year_end):
        
        yr2 = yr1 + 1
        file_name = "data/nba_advanced_stats/"
        file_name += "nba_advanced_" + str(yr1) + "_" + str(yr2) + ".csv"
        
        print(file_name)
        data = pd.read_csv(file_name)
        
        cur_players = list(data.Player)
        
        player_list.extend(cur_players)
        
    player_list = list(set(player_list))
    
    return player_list





nba_player_names = get_nba_player_names(2009, 2019)




# Importing the dataset
data = pd.read_csv("data/college_stats/final_CBB_data.csv")

college_data = data.copy()

# get y
future_stats, dropped_player_indeces = get_y(data, "WS")


# drop players from X if they didnt qualify for y
for index, player, year in dropped_player_indeces:
    print("dropping ", player)
    college_data = college_data.drop([index], axis=0)
 
    

college_stat_type = "TRB"
    
college_stat = college_data[college_stat_type]
    
    

import seaborn as sns
import matplotlib.pyplot as plt




ax = sns.scatterplot(x=college_data[college_stat_type], y=future_stats)
ax.set(xlabel="college " + college_stat_type , ylabel='NBA WS')







# Draw a nested barplot to show survival for class and sex
g = sns.catplot(x="college AST", y="pred", hue="player", data=graphData,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("Win Shares")
    
sns.set(style="whitegrid")

    
    
    
    
    
#####################  get correlation ############################
    
from scipy.stats import pearsonr

columns = list(college_data.columns)
diff = lambda l1, l2: [x for x in l1 if x not in l2]
columns = diff(columns, ["spacer1", "playerName", "Season", "School", "Conf", "position"])


from sklearn.impute import SimpleImputer



corrs = []
ps = []
for coll_stat in columns:
    print(coll_stat)
    stat = np.array(college_data[coll_stat])
    
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")


    imputer = imputer.fit(stat.reshape(-1, 1))
    stat = imputer.transform(stat.reshape(-1, 1))
    stat = stat.reshape(1228)
    
    print(stat)
    correlation = pearsonr(stat, future_stats)


    corrs.append(correlation[0])
    ps.append(correlation[1])
    
    
    
predDF = pd.DataFrame()
predDF["college stat name"] = columns
predDF["correlation"] = corrs
predDF["ps"] = ps

predDF = predDF.sort_values(by="correlation", ascending=False)
    
#####################  /get correlation ############################
    
    
    
    
    
    
    
    
    



    
graphData = pd.DataFrame()

graphData["college AST"] = college_stat
graphData["NBA WS"] = future_stats

    
    
import seaborn as sns
sns.set(style="ticks")

df = sns.load_dataset("iris")
sns.pairplot(df, hue="species")
    
    

    
    
    
    
    
    
    
    
    
    