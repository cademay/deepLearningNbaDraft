   #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:50:37 2019

@author: cademay
"""


from bs4 import BeautifulSoup
from urllib.request import urlopen
import csv
from time import sleep
import pandas as pd
import numpy as np
import re

draftURL = "https://www.cbssports.com/nba/draft/prospect-rankings/"

player_data_class = "cell-bold-text cell-player"

html = urlopen(draftURL)
soup = BeautifulSoup(html)

player_names = []
          
player_rows = soup.find_all(class_=re.compile(player_data_class))

# extract player names
for row in player_rows:
    player_url = row.a
    if player_url:
        player = player_url.getText()
        player_names.append(player)
        
# remove duplicates
player_names = list(set(player_names))

# save names
namesDF = pd.DataFrame(player_names, columns=["Player"])
namesDF.to_csv(path_or_buf="draft_prospects_names_2019.csv")
