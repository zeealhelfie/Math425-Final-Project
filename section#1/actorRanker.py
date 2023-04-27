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
        
        ls=line.rstrip("\n").split("/")
        ls.pop(0) # Remove first item since it's a movie title, not an actor
        actors.update(set(ls))

f.close()

# sort list of all actors alphabetically
actors=sorted(list(actors))
#print(actors)

n = len(actors)
# Create adjacency matrix
G = zeros((n,n))

with open("data/top250movies.txt","r", encoding='UTF-8') as f:
    for line in f:
        ls=line.rstrip("\n").split("/")
        # Convert actors/names to their IDs/indices in the sorted list
        lIDs = map(actors.index, ls)
        ls.pop(0)
        row = [] # Convert lID to iterable/enumerable array
        for idx, i in enumerate(lIDs):
            row.append(i)
            
        for idx, i in enumerate(row):
            # Link node j to node i,
            # i.e. increment weight of G_{ij} for each j in the rest of the row
            
            # i represents the higher billed actor,
            #   row[idx+1:] are the lower billed actors.
            G[i, row[idx+1:]] += 1.0

f.close()

# Change epsilon to get different results
netw=pageRank(G,actors,0.69,1e-6,100)  
result = netw.powermethod()

r=open("actorResults.txt", "w")
for i in result:
    r.write(f"{i[0]}, [ {i[1]} ]\n")
r.close()