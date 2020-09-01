# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 08:45:20 2019

@author: Samuel Fisher, Intern
Johns Hopkins University Applied Physics Laboratory
"""

#Display who won and add to win counter
def whoWin(x,End,Xwin,Owin): 
    Xwin = 0
    Owin = 0
    if x == 1:
        End.configure(text="Player 1 has won!", background = 'white')
        Xwin = 1
    elif x == 2:
        End.configure(text="Player 2 has won!", background = 'white')
        Owin = 1
    else:
        End.configure(text="Nobody Wins", background = 'white')
    gameover = 1
    L = [Xwin,Owin,gameover]
    return L

#Check if there is a three in a row
#If there is a win, a display which team one and count that win
def checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill): 
    if place[1] == place[0] and place[0] == place[2] and place[1] != 0:
        print ("Player",place[1]," wins")
        return whoWin(place[1],End,Xwin,Owin)
    if place[0] == place[3] and place[0] == place[6] and place[0] != 0:
        print ("Player",place[0]," wins")
        return whoWin(place[0],End,Xwin,Owin)
    if place[0] == place[4] and place[0] == place[8] and place[0] != 0:
        print ("Player",place[0]," wins")
        return whoWin(place[0],End,Xwin,Owin)
    if place[1] == place[4] and place[1] == place[7] and place[1] != 0:
        print ("Player",place[1]," wins")
        return whoWin(place[1],End,Xwin,Owin)
    if place[2] == place[4] and place[2] == place[6] and place[2] != 0:
        print ("Player",place[2]," wins")
        return whoWin(place[2],End,Xwin,Owin)
    if place[2] == place[5] and place[2] == place[8] and place[2] != 0:
        print ("Player",place[2]," wins")
        return whoWin(place[2],End,Xwin,Owin)
    if place[3] == place[4] and place[3] == place[5] and place[3] != 0:
        print ("Player",place[3]," wins")
        return whoWin(place[3],End,Xwin,Owin)
    if place[6] == place[7] and place[8] == place[6] and place[6] != 0:
        print ("Player",place[6]," wins")
        return whoWin(place[7],End,Xwin,Owin)
    tie = 1
    for i in place:
        if i == 0:
            tie = 0
    if tie == 1:
        return whoWin(3,End,Xwin,Owin)
        
    return [0,0,0]

#Check who won without calling whoWin
#Necessary for MiniMax
def checkWin2(place):
    if place[1] == place[0] and place[0] == place[2] and place[1] != 0:
        return place[1]
    if place[0] == place[3] and place[0] == place[6] and place[0] != 0:
        return place[0]
    if place[0] == place[4] and place[0] == place[8] and place[0] != 0:
        return place[0]
    if place[1] == place[4] and place[1] == place[7] and place[1] != 0:
        return place[1]
    if place[2] == place[4] and place[2] == place[6] and place[2] != 0:
        return place[2]
    if place[2] == place[5] and place[2] == place[8] and place[2] != 0:
        return place[2]
    if place[3] == place[4] and place[3] == place[5] and place[3] != 0:
        return place[3]
    if place[6] == place[7] and place[8] == place[6] and place[6] != 0:
        return place[6]
    tie = 1
    for i in place:
        if i == 0:
            tie = 0
    if tie == 1:
        return 0
        
    return None

################################################
# checkWinPos
# 
# Checks for winning positions, will block a win
################################################
def checkWinPos(place):
    win_states = [[0, 1, 2],
                  [0, 3, 6],
                  [0, 4, 8],
                  [1, 4, 7],
                  [2, 4, 6],
                  [2, 5, 8],
                  [3, 4, 5],
                  [6, 7, 8]]

    for value in [2, 1]:
       for win in win_states:
           L = [place[win[0]], place[win[1]], place[win[2]]]
           if (L == [value, value, 0] or
               L == [value, 0, value] or
               L == [0, value, value]):
               index = win[L.index(min(L))]
               return index
    return None