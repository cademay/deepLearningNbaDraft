#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:15:04 2019

@author: cademay
"""

#import glob
#allfiles = glob.glob("data/nba_advanced_stats/*")

from bs4 import BeautifulSoup
from urllib.request import urlopen
import csv
from time import sleep
import pandas as pd
import numpy as np

# advanced stats from : https://www.basketball-reference.com/leagues/NBA_2019_advanced.html
# predict year 6 or 7
templateURL = "https://www.sports-reference.com/cbb/players/FIRSTNAME-LASTNAME-1.html"

#players = ["John Stockton*\stockjo01","James Harden\hardeja01", "Shaquille O'Neal*\onealsh01"]


def get_nba_player_names():

    player_list = []
    
    for yr1 in range(2000,2019):
        
        yr2 = yr1 + 1
        file_name = "data/nba_advanced_stats/"
        file_name += "nba_advanced_" + str(yr1) + "_" + str(yr2) + ".csv"
        data = pd.read_csv(file_name)
        
        cur_players = list(data.Player)
        
        player_list.extend(cur_players)
        
    player_list = list(set(player_list))
    
    return player_list


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
    

def extractPlayerData(soup):

    cmHeight, kgWeight, position = extract_height_weight_and_position(soup)
    
    # get stats 
    rows = soup.findAll('tr')[1:-1]
    player_stats = [[td.getText() for td in rows[i].findAll('td')] for i in range(len(rows))]
    finalYearStats = player_stats[len(player_stats)-1] 
    
    if len(finalYearStats) != 28:
        raise("data too old: no ORB / DRB")
    
    # get draft year
    finalSeasonRow = rows[len(rows) - 1]
    finalSeasonYearNumber = finalSeasonRow.find("th").getText() #draftYr
    
    
    # get years in College
    collegeYearsPlayed = len(rows) 
    
    finalYearStats.insert(0, finalSeasonYearNumber)
    finalYearStats.append(collegeYearsPlayed)
    finalYearStats.append(cmHeight)
    finalYearStats.append(kgWeight)
    finalYearStats.append(position)
    
    return finalYearStats
    
    


player_list = get_nba_player_names()
player_list = player_list[982:]
firstDataPoint = True
allData = []

for player in player_list:
    
    firstName, lastName = getName(player)
 
    newURL = templateURL.replace("FIRSTNAME", firstName)
    newURL = newURL.replace("LASTNAME", lastName)
    
    print(newURL)
    
    try:
        
        html = urlopen(newURL)
        soup = BeautifulSoup(html)
        
        playerData = extractPlayerData(soup)
        
        playerData.insert(0, player) # player's name
        
        if firstDataPoint:
            headers = [th.getText() for th in soup.findAll('tr', limit=2)[0].findAll('th')]
            headers.insert(0, "playerName")
            headers.extend(["yearsInCollege","heightInCm","weightInKg","position"])
            firstPlayer = False
            
        allData.append(playerData)
        
        sleep(1.5)
 
    except: 
        print("player not found: ", player)


data = pd.DataFrame(allData, columns = headers)


data.to_csv(path_or_buf = "college_basketball_data1.csv")


