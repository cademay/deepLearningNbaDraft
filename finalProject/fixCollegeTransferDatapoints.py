#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 00:42:13 2019

@author: cademay
"""

from bs4 import BeautifulSoup
from urllib.request import urlopen
import csv
from time import sleep
import pandas as pd
import numpy as np
import math

# advanced stats from : https://www.basketball-reference.com/leagues/NBA_2019_advanced.html
# predict year 6 or 7
templateURL = "https://www.sports-reference.com/cbb/players/FIRSTNAME-LASTNAME-1.html"

df = pd.read_csv("data/college_stats/college_basketball_data.csv")

seasons = list(df.Season)




def extract_height_weight_and_position(soup):
    
    paragraphs = soup.findAll("p")

    # ignore nickname paragraph
    posIndex = 0 if len(paragraphs[0]) > 1 else 1

        
    # get position (eg. guard, center, etc)
    positionParagraph = paragraphs[posIndex]
    position = list(positionParagraph.children)[2].strip()

    # get height, weight in cm, kg

    sizeParagraph = paragraphs[posIndex + 1]
    sizeChildren = list(sizeParagraph.children) # looks like this: '\xa0(196cm,\xa098kg) '
    size = sizeChildren[3].replace("\xa0", "")
    size = size.replace("(", "") # now this : '196cm,98kg) '
    
    cmHeight = size[0:size.index("cm")]  
    kgWeight = size[size.index(",") + 1 : size.index("kg")]
    
    return (cmHeight, kgWeight, position)



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


def getUpdates(playerDatapoint):
    
    playerFullName = playerDatapoint[0]
    
    firstName, lastName = getName(playerFullName)
 
    newURL = templateURL.replace("FIRSTNAME", firstName)
    newURL = newURL.replace("LASTNAME", lastName)
    
    print(newURL)
    
    # SOUP        
    html = urlopen(newURL)
    soup = BeautifulSoup(html)

    cmHeight, kgWeight, position = extract_height_weight_and_position(soup)

    # get stats 
    career = soup.findAll('tbody')[0].findAll('tr')
    career_stats = [[td.getText() for td in career[i].findAll('td')] for i in range(len(career))]
    finalYearStats = career_stats[len(career_stats) - 1] 
    
        
    if len(finalYearStats) != 28:
        raise("data too old: no ORB / DRB")
    
    seasonNumbers = soup.findAll('tbody')[0].findAll('th')
    finalSeasonYearNumber = seasonNumbers[-1].getText()
    
    # get years in College
    collegeYearsPlayed = len(career_stats) 
    
    finalYearStats.insert(0, finalSeasonYearNumber)
    finalYearStats.append(collegeYearsPlayed)
    finalYearStats.append(cmHeight)
    finalYearStats.append(kgWeight)
    finalYearStats.append(position)
    
    
    return finalYearStats
        




for index in range(len(df)):

    
    #curSeason = df.at[len(df)-10, "Season"]
    #print(curSeason)
    
    season = df.at[index, "Season"]
    conf = df.at[index, "Conf"]
    
    if not isinstance(season, str) and not isinstance(conf, str):
        # update transfer student info
        
        seasonless_player = (list(df.iloc[index, :]))
        print(seasonless_player)
        playerDatapoint = seasonless_player
        
        try:
            updatedStats = getUpdates(seasonless_player)
            
            playerName = seasonless_player[0]
            updatedStats.insert(0, playerName) # player's name
            
            df.iloc[index, :] = updatedStats
            
           
            
            print(updatedStats)
            print("") 
            sleep(1.5)
        except: 
            print("player not found: ", playerName)
    
        #df.at[index, "Season"] = updatedSeason
        #df.at[index, "Conf"] = updatedConference
    #if curSeason == "nan" :
    #    print(df.iloc[i, :])
        
#df.to_csv(path_or_buf = "test.csv")
