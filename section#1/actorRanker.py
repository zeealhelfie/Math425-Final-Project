#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 2023
Based on Boateng's web_stanford_pagerank.py
@author: henryboateng, Cameron Yee
"""
from imp import reload
from numpy import array, identity, ones, nonzero, zeros
from numpy import linalg as LA

from PageRank_module import pageRank

actors=set()
with open("data/top250movies.txt","r", encoding='UTF-8') as f:
    for line in f:
        
        #print(line)
        ls=line.rstrip("\n").split("/")
        ls.pop(0)#remove first item since it's a movie title, not an actor
        #print(set(ls))
        actors.update(set(ls))

f.close()

n = len(actors)
# Create adjacency matrix
G = zeros((n,n))

with open("data/top250movies.txt","r", encoding='UTF-8') as f:
    for line in f:
        ls=line.rstrip("\n").split("/")
        lIDs = map(list(actors).index, ls) #IDs on the line
        ls.pop(0)
        row = [] # row indices of IDS
        for i in lIDs:
            row.append(i)
            
        j = row[0] # First element is column index
        # Link page j to page i, i.e. set G_{ij} to 1 
        G[row[1:],j] += 1.0

f.close()  

netw=pageRank(G,actors,0.85,1e-6,15)  
result = netw.linsolve()

r=open("actorResults.txt", "x")
for i in result:
    r.write(f"{i[0]}, [ {i[1]} ]\n")
r.close()
