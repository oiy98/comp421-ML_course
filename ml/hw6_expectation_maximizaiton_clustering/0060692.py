#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import scipy.spatial as spa
import random
import math
from numpy.linalg import inv, det


# In[2]:


def sortd(covariance):
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    return values[order], vectors[:,order]


# In[3]:


def update_centroids(memberships, X):
    if memberships is None:
        # initialize centroids
        centroids = X[np.random.choice(range(N), K),:]
    else:
        # update centroids
        centroids = np.vstack([np.mean(X[memberships == k,], axis = 0) for k in range(K)])
    return(centroids)

def update_memberships(centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return(memberships)


# In[4]:


def plot_current_state(centroids, memberships, X):
    cluster_colors = np.array(["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"])
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    if memberships is None:
        plt.plot(X[:,0], X[:,1], ".", markersize = 10, color = "black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 10, 
                     color = cluster_colors[c])
    for c in range(K):
        plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize = 12, 
                 markerfacecolor = cluster_colors[c], markeredgecolor = "black")
    plt.xlabel("x1")
    plt.ylabel("x2")


# In[5]:


def KNN(X):
    centroids = None
    memberships = None
    iteration = 2
    while True:
        print("Iteration#{}:".format(iteration))

        old_centroids = centroids
        centroids = update_centroids(memberships, X)
        if np.alltrue(centroids == old_centroids):
            break
        else:
            plt.figure(figsize = (12, 6))    
            plt.subplot(1, 2, 1)
            plot_current_state(centroids, memberships, X)

        old_memberships = memberships
        memberships = update_memberships(centroids, X)
        if np.alltrue(memberships == old_memberships):
            plt.show()
            break
        else:
            plt.subplot(1, 2, 2)
            plot_current_state(centroids, memberships, X)
            plt.show()

        iteration = iteration + 1

    return centroids , memberships


# In[13]:


def EM_alg(X, K, Mi, Ri, means, cov):
    N= X.shape[0]
    D=X.shape[1]
    Member = np.zeros((K, D))
    Center = np.zeros((K, D, D))
    pi = np.ones(K) / K
    if Ri.all() == None:
        R = np.zeros((N, K))
    else:
        R = Ri
    for k in range(K):
        Member[k] = X[np.random.choice(N)]
        Center[k] = np.eye(D)
    if Mi.all() != None:
        Member = Mi
    pdfs = np.zeros((N, K))
    for i in range(100):
        for k in range(K):
            for n in range(N):
                pdf=multivariate_normal.pdf(X[n], Member[k], Center[k])
                pdfs[n,k] = pi[k]*pdf
        for k in range(K):
            for n in range(N):
                Sumpdf=(np.sum(pdfs[n,:]))
                R[n,k] = pdfs[n,k] /Sumpdf
        for k in range(K):
                x = R[:,k].sum()
                pi[k] = x / N
                Member[k] = np.dot(R[:,k],(X)) / x
                Center[k] = np.sum(R[n,k]*np.outer(X[n] - Member[k], X[n] - Member[k]) for n in range(N)) / x + np.eye(D)*10e-4
    plt.scatter(X[:,0], X[:,1])
    grph = plt.subplot(111)
    for i in range(K):
        mem = Member[i]
        cen = Center[i]
        vals = sortd(cen)[0]
        width=(2 * 2 * np.sqrt(vals))[0]
        height= (2 * 2 * np.sqrt(vals))[1]
        ellipse = Ellipse(xy=(mem[0],mem[1]),width=width,height=height,color='black', ls='--')
        ellipse.set_facecolor('none')
        grph.add_artist(ellipse)
    plt.show()
    print (Member)
    return R


# In[7]:


def hot_encode(R):
    ans=np.zeros((300,5))
    for i in range(len(R)):
        if R[i].all()==0:
            ans[i,:]=[1,0,0,0,0]
        if R[i].all()==1:
            ans[i,:]=[0,1,0,0,0]
        if R[i].all()==2:
            ans[i,:]=[0,0,1,0,0]
        if R[i].all()==3:
            ans[i,:]=[0,0,0,1,0]
        if R[i].all()==4:
            ans[i,:]=[0,0,0,0,1]
    return ans


# In[8]:


# initializing class parameters
class_means = np.array([[+2.5, +2.5], [-2.5, +2.5], [-2.5, -2.5],[-2.5, +2.5],[0, 0]])
class_deviations = np.array([[[0.8, -0.6], [-0.6, 0.8]],[[0.8, 0.6], [0.6, 0.8]], [[0.8, -0.6], [-0.6, 0.8]],[[0.8, 0.6], [0.6, 0.8]],[[1.6, 0], [0, 1.6]]])
class_sizes = [50,50,50,50,100]


# In[9]:


np.random.seed(421)
# sample size
N = 300
# cluster count
K = 5


# In[10]:


X1 = np.random.multivariate_normal(np.array([+2.5, +2.5]), np.array([[0.8, -0.6], [-0.6, 0.8]]), 50)
X2 = np.random.multivariate_normal(np.array([-2.5, +2.5]), np.array([[0.8, 0.6], [0.6, 0.8]]), 50)
X3 = np.random.multivariate_normal(np.array([-2.5, -2.5]), np.array([[0.8, -0.6], [-0.6, 0.8]]), 50)
X4 = np.random.multivariate_normal(np.array([+2.5, -2.5]), np.array([[0.8, 0.6], [0.6, 0.8]]), 50)
X5 = np.random.multivariate_normal(np.array([0, 0]), np.array([[1.6, 0], [0, 1.6]]), 100)
X = np.vstack((X1, X2, X3, X4,X5))


# In[11]:


plt.plot(X[:,0],X[:,1],'o')
M, R = KNN(X)


# In[ ]:


R=hot_encode(R)
R = EM_alg(X, 5, M, R, class_means, class_deviations)


# In[ ]:




