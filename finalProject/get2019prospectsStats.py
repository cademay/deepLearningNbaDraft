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
import re
# advanced stats from : https://www.basketball-reference.com/leagues/NBA_2019_advanced.html
# predict year 6 or 7
templateURL = "https://www.sports-reference.com/cbb/players/FIRSTNAME-LASTNAME-1.html"

#players = ["John Stockton*\stockjo01","James Harden\hardeja01", "Shaquille O'Neal*\onealsh01"]


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
    
def transfer_helper(soup):

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
    

def check_transfer(soup):
    
    stats = transfer_helper(soup)
    
    if stats[0] == '' and stats[2] == '':
        return 1
    
    return 0
    
def extractData(soup):
    

    cmHeight, kgWeight, position = extract_height_weight_and_position(soup)

    # get stats 
    career = soup.findAll('tbody')[0].findAll('tr')
    career_stats = [[td.getText() for td in career[i].findAll('td')] for i in range(len(career))]
    finalYearStats = career_stats[len(career_stats) - 1] 
    
    
    if len(finalYearStats) != 28:
        print(finalYearStats)
        print(len(finalYearStats))
        raise("data too old: no ORB / DRB")
    
    seasonNumbers = soup.findAll('tbody')[0].findAll('th')
    finalSeasonYearNumber = seasonNumbers[-1].getText()
    
    # get years in College
    collegeYearsPlayed = len(career_stats) 
    
    # check if transferred schools or not
    transferred = check_transfer(soup)
    
    finalYearStats.insert(0, finalSeasonYearNumber)
    finalYearStats.append(collegeYearsPlayed)
    finalYearStats.append(cmHeight)
    finalYearStats.append(kgWeight)
    finalYearStats.append(position)
    finalYearStats.append(transferred)
    
    
    return finalYearStats
        

def getName2019(player):
    
    player = player.replace(".", "") # eg J.J. Redick
    player = player.replace("'", "") # Shaquille O'Neal*\onealsh01"
    player = player.replace("*", "") # Shaquille O'Neal*\onealsh01"
    
    
    spaceIndex = player.index(" ")
    
    player = player.lower()
        
    firstName = player[0 : spaceIndex]
    lastName = player[spaceIndex + 1 : ].strip()
    
    return (firstName, lastName)


player_list = pd.read_csv("data/college_stats/draft_prospects_names_2019.csv")
player_list = list(player_list.Player)

firstDataPoint = True
allData = []

for player in player_list:
    
    firstName, lastName = getName2019(player)

    
    newURL = templateURL.replace("FIRSTNAME", firstName)
    newURL = newURL.replace("LASTNAME", lastName)
    
    print(newURL)
    
    try:
        
        html = urlopen(newURL)
        soup = BeautifulSoup(html)

        playerData = extractData(soup)
        
        playerData.insert(0, player) # player's name
        
        if firstDataPoint:
            headers = [th.getText() for th in soup.findAll('tr', limit=2)[0].findAll('th')]
            headers.insert(0, "playerName")
            headers.extend(["yearsInCollege","heightInCm","weightInKg",
                            "position", "transferredSchools"])
    
            firstPlayer = False
            
        allData.append(playerData)
        
        sleep(3)
 
    except: 
        print("player not found: ", player)
        



data = pd.DataFrame(allData, columns = headers)
data = data.drop([39], axis=0)




new_urls = [
        "https://www.sports-reference.com/cbb/players/jared-harper-12.html",
        "https://www.sports-reference.com/cbb/players/cameron-johnson-4.html",
        "https://www.sports-reference.com/cbb/players/justin-james-2.html",
        "https://www.sports-reference.com/cbb/players/justin-robinson-3.html",
        "https://www.sports-reference.com/cbb/players/simisola-shittu-1.html",
        "https://www.sports-reference.com/cbb/players/kezie-okpala-1.html",
        "https://www.sports-reference.com/cbb/players/nicolas-claxton-1.html",
        "https://www.sports-reference.com/cbb/players/chris-clemons-2.html",
        "https://www.sports-reference.com/cbb/players/aubrey-dawkins-1.html",
        "https://www.sports-reference.com/cbb/players/bol-bol-1.html",
        "https://www.sports-reference.com/cbb/players/amir-coffey-1.html",
        "https://www.sports-reference.com/cbb/players/john-konchar-1.html",
        "https://www.sports-reference.com/cbb/players/ky-bowman-1.html",
        "https://www.sports-reference.com/cbb/players/reggie-perry-2.html",
        "https://www.sports-reference.com/cbb/players/kevin-porterjr-1.html",
        "https://www.sports-reference.com/cbb/players/charlie-brown-2.html",
        "https://www.sports-reference.com/cbb/players/tyus-battle-1.html",
        "https://www.sports-reference.com/cbb/players/lindell-wigginton-1.html",
        "https://www.sports-reference.com/cbb/players/obadiah-toppin-1.html",
        "https://www.sports-reference.com/cbb/players/jontay-porter-1.html",
        "https://www.sports-reference.com/cbb/players/kerwin-roachjr-1.html",
        "https://www.sports-reference.com/cbb/players/temetrius-morant-1.html",
        "https://www.sports-reference.com/cbb/players/zach-norvelljr-1.html",
        "https://www.sports-reference.com/cbb/players/shamorie-ponds-1.html",
        "https://www.sports-reference.com/cbb/players/juwan-morgan-1.html",
        "https://www.sports-reference.com/cbb/players/charles-matthews-1.html",
        "https://www.sports-reference.com/cbb/players/jaylen-hands-1.html"
        
        ]




new_data = []
for url in new_urls:
    
    print(url)
    
    try:
        
        html = urlopen(url)
        soup = BeautifulSoup(html)

        title = soup.find("title").getText()
        
        player = re.findall("(.*)[ ]{1}College Stats", title)[0]
        

        print("soup acquired, preparing to extract")
        playerData = extractData(soup)
        
        playerData.insert(0, player) # player's name
        
        if firstDataPoint:
            headers = [th.getText() for th in soup.findAll('tr', limit=2)[0].findAll('th')]
            headers.insert(0, "playerName")
            headers.extend(["yearsInCollege","heightInCm","weightInKg",
                            "position", "transferredSchools"])
    
            firstPlayer = False
            
        new_data.append(playerData)
        
        sleep(3)
        print()
        
    except: 
        print("player not found: ", url)
        
        
        




df =pd.DataFrame(new_data, columns=headers) 
df.to_csv(path_or_buf = "college_basketball_data_2019_draft_prospects3.csv")







data.to_csv(path_or_buf = "college_basketball_data_2019_draft_prospects1.csv")


