

'''
This program computes Page Rank using NumPy library
It takes a stochastic adjacency matrix M as input,
and returns a vector r of page rankes.

Author: Raji Ghawi
Date 21/9/2015
'''

import numpy as np
from numpy import linalg as LA

# program parameters:
# epsilon: convergence threshold, 
# i.e., power iteration stops when the difference in r values becomes less than epsilon
eps = 0.0000001
# maxNbIters: Maximum number of iterations
maxNbIters = 1000
# beta: probability to follow a link at random. 
# 1-beta: probability to jump to some random page. 
beta = 0.8

# Stochastic adjacency matrix
M = np.matrix([[0.0, 0.0, 1.0], [0.5, 0, 0], [0.5, 1.0, 0]])
print(M)

n = M.shape[0]
rl = []
for i in range(n):    
    rl.append(1.0/n)
r = np.array(rl)
r.shape = (n,1)

A_ = (1.0/n)*np.ones((n, n))
A = beta * M + (1.0-beta)*A_
print(A)


dif = None
nbIter = 0
while (dif is None or dif>eps) and nbIter<maxNbIters:
    r1 = A * r
    #print(r1)
    dif = LA.norm(r1 - r, 1)    
    #print(dif)
    r =r1
    nbIter += 1

print('')
print('Final ranks:')
print(r.transpose())
print('#iters = '+str(nbIter))

