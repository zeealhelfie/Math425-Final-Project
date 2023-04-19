"""
Created on Sun Feb 19 21:13:22 2023

@author: hboateng
"""

from imp import reload
from numpy import array, identity, ones, nonzero, zeros
from numpy import linalg as LA

def sortPageRank(prdict):
    return sorted(prdict.items(), key = lambda item: item[1], \
                  reverse=True)
    
class pageRank():
    def __init__(self,G,nodelist,eps,tol,maxiter):
        
        n = G.shape[0]
        for i in range(n): # normalize columns of A
            ci = LA.norm(G[:,i],1)
            if ci > 0:
                G[:,i]=G[:,i]/ci
            else: # adjustment for a column of zeros
                G[:,i]=ones((n,))/float(n)           
            
        self.G       = G         # normalized matrix        self.nodes   = nodelist  # list of node labels
        self.eps     = eps       # probability of jumping to a link on page
        self.size    = G.shape[0]# size of matrix
        self.tol     = tol # tolerance for power method
        self.maxiter = maxiter # maximum number of iterations for power method
        if not nodelist: # list of node labels
            self.nodes = [k for k in range(self.size)]
        else:
            self.nodes = nodelist
            
    def linsolve(self):
        n = self.size
        x = LA.solve(identity(n)-self.eps*self.G,\
                        (1.0-self.eps)/n*ones((n,1)))
        return sortPageRank({k:float(v) for (k,v) in zip(self.nodes,x)})
    
    def marchmadness(self,tourneyteams):
        n = self.size
        x = LA.solve(identity(n)-self.eps*self.G,\
                        (1.0-self.eps)/n*ones((n,1)))
        tteams = sorted(tourneyteams)
        
        stteams = sortPageRank({k:float(x[self.nodes.index(k)]) for k in tteams})
        return stteams, sortPageRank({k:float(v) for (k,v) in zip(self.nodes,x)})
    
    def eigensolve(self):
        n = self.size
        P = self.eps*self.G + (1.0-self.eps)/self.size*ones((n,n))
        evals, evecs = LA.eig(P)
        
        idx = evals.argsort()[::-1]   
        evals = evals[idx]
        evecs = evecs[:,idx]
        
        # return normalized eigenvector corresponding to eval = 1
        x = abs(evecs[:,0])/LA.norm(evecs[:,0],1)
        return sortPageRank({k:v for (k,v) in zip(self.nodes,x)})
    
    def powermethod(self):
        n = self.size
        
        # Get sparse G (as a list)
        
        #list of lists of index of nonzero elements in each row
        nzre = [nonzero(self.G[k,:]>0) for k in range(n)] 
        
        #list of vectors of nonzero elements in each row
        nzv = [self.eps*self.G[k,nzre[k]] for k in range(n)]
        
        #for k in range(n):
         #   print(f"nzre = {nzre[k]}, nzv = {nzv[k]}")
        
        oeps = (1.0-self.eps)/n
        x = ones((n,1))/float(n) # initial vector

        
        xn1 = LA.norm(x,1)
        ntol = xn1
        niter = 0
        while ntol > self.tol and niter < self.maxiter :
            xold = x       
            for k in range(n):
                x[k] = nzv[k]@x[nzre[k]] + oeps
                
            xn1  = LA.norm(x,1)
            x    = x/xn1
            ntol = LA.norm(x-xold,1)
            
            niter += 1
            print(f"n = {niter}, ntol = {ntol}, x = {x}")
            
        return sortPageRank({k:float(v) for (k,v) in zip(self.nodes,x)})


G1 = array([ [0, 0.0, 0.0, 0.0], [1.0, 0, 1.0, 0], [1.0, 0, 0, 0],  
           [1.0,1.0, 1.0, 0]])
netw1=pageRank(G1,['1','2','3','4'],0.85,1e-15,15)

G2 = array([ [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0], [1.0, 0, 1.0, 0],  
           [1.0,1.0, 1.0, 1.0]])
netw2=pageRank(G2,['1','2','3','4'],0.85,1e-15,15)

G3 = array([ [0, 1.0, 1.0, 0, 0, 0], 
             [1.0, 0, 0, 1.0, 1.0, 0],
             [1.0, 0, 0, 0, 0, 1.0],
             [0, 1.0, 0, 0, 0, 0],
             [0, 1.0, 0, 1.0, 0, 1.0],
             [0, 0, 1.0, 0, 1.0, 0]])

netw3=pageRank(G3,['A','B','C','D','E','F'],0.85,1e-8,10)
        
 
