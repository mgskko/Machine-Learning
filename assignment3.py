#!/usr/bin/env python
# coding: utf-8

# In[79]:


import pandas as pd
import numpy as np
import math
import csv
import random
import sys


# In[80]:


df = pd.read_csv('https://it.yonsei.ac.kr/adslab/faculty/data_mining/assignment2_input.txt', sep = '\t',
                     names=['1', '2', '3','4','5','6','7','8','9','10','11','12'])


# In[81]:


def distance(A, B):
  distance = 0
  for i in range(len(A)):
    distance += (A[i] - B[i]) ** 2
  return distance


# In[82]:


def kMedoids(D, k, tmax=100):
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs,cs = np.where(D==0)
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r,c in zip(rs,cs):
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(
            len(invalid_medoid_inds)))

    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    Mnew = np.copy(M)

    C = {}
    for t in range(tmax):
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # return results
    return M, C


# In[83]:


for i in range(0,500):
    D += distance(df.values[i],df.values[i])
# split into 2 clusters
M, C = kMedoids(D, 10)


# In[108]:


with open('assignment3_output.txt','w',encoding='UTF-8') as f:
    for i in range(0, 10):
        f.write(str(len(C[i])) + ':' + str(C[i].tolist()) +'\n')
f.close()


# In[ ]:




