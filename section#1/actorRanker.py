#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 2023
Based on Boateng's web_stanford_pagerank.py and march_madness.py files
@author: henryboateng, math425 group2
"""
from imp import reload
from numpy import array, identity, ones, nonzero, zeros, eye
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

actorsSorted=sorted(list(actors))
#print(actorsSorted)

n = len(actors)
# Create adjacency matrix
G = zeros((n,n))

with open("data/top250movies.txt","r", encoding='UTF-8') as f:
    for line in f:
        ls=line.rstrip("\n").split("/")
        lIDs = map(actorsSorted.index, ls) # IDs on the line
        ls.pop(0)
        row = [] # row indices of IDS
        for idx, i in enumerate(lIDs):
            #print(i, ls[idx])
            row.append(i)
            
        for idx, i in enumerate(row):
            # First element is column index
            # Link node j to node i, i.e. increment weight of G_{ij} 
            G[i, row[idx+1:]] += 1.0
            #print(ls[idx], G[i, row[idx+1:]])

f.close()

netw=pageRank(G,actorsSorted,0.85,1e-6,15)  
result = netw.linsolve()

r=open("actorResults.txt", "w")
for i in result:
    r.write(f"{i[0]}, [ {i[1]} ]\n")
r.close()
