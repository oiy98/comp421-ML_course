#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.linalg import solve
from scipy import stats
from sklearn.metrics import confusion_matrix
import scipy.spatial.distance as dt


# In[38]:


# read data into memory

train_imags= np.genfromtxt("hw07_training_images.csv",delimiter=",")
train_labels= np.genfromtxt("hw07_training_labels.csv",delimiter=",")
test_imags= np.genfromtxt("hw07_test_images.csv",delimiter=",")
test_labels= np.genfromtxt("hw07_test_labels.csv",delimiter=",")

#x and y values
X_train=train_imags[:,]
X_test=test_imags[:,]
Y_train=train_labels[:,]
Y_test=test_labels[:,]


# In[39]:


N=len(Y_train)
D=np.shape(X_train)[1]
K=max(Y_train)
K=np.int(K)


# In[40]:


#MEANS
class_means=[]
for i in range(1,K+1):
    class_means.append(np.mean(X_train[Y_train==i,],axis=0))
class_means=np.array(class_means)


# In[41]:


X_train_div_mean=[]
for i in range(N):
    X_train_div_mean.append(X_train[i,:]-class_means[np.int(Y_train[i])-1,:])
X_train_div_mean= np.array(X_train_div_mean)


# In[42]:


total_mean=np.mean(class_means,axis=0)


# In[43]:


#class scatter matrix calculation
def scatter(x):
    ans=np.zeros((D,D))
    for i in range(N):
        if (Y_train[i]==x):
            ans=ans+np.dot(X_train_div_mean[i,:],np.transpose(X_train_div_mean[i,:]))
    return ans


# In[44]:


#within class scatter matrix
def scatterw():
    Sw=np.zeros((D,D))
    for i in range(K):
        index = np.where(Y_train == i+1)[0]
        class_scatter = np.zeros((D,D))
        mv = class_means[i].reshape(D, 1)
        for row in index:
            mrow =  X_train[row].reshape(D, 1)
            class_scatter += (mrow - mv).dot((mrow - mv).T)
        Sw += class_scatter
    return Sw


# In[45]:


#between clss scatter
def scatterb():
    ans=np.zeros((D,D))
    for i in range(K):
        n = len(np.where(Y_train == i+1)[0])
        mv = class_means[i].reshape(D,1)
        ovm = total_mean.reshape(D,1)
        ans += n * (mv - ovm).dot((mv-ovm).T)
    return ans


# In[46]:


within=scatterw()
between=scatterb()


# In[47]:


#singularity
for d in range(D):
    within[d,d]=within[d,d]+1e-10


# In[48]:


#eigen values and eigen vectors
within_inversed=np.linalg.inv(within)
ques= np.dot(within_inversed,between)
values,vectors=la.eigh(ques)


# In[49]:


def calculateZ(X,R):
    wid=vectors[:,0:R]
    return(np.dot(X,wid))


# In[50]:


Z_train=calculateZ(X_train,2)
Z_test= calculateZ(X_test,2)


# In[62]:


#plot 2 dim projection
point_colors = ["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00",]
for i in range(N):
    plt.scatter(Z_train[i,0],-Z_train[i,1],color=point_colors[np.int(Y_train[i])],s=5)


# In[71]:


for i in range(len(Y_test)):
    plt.scatter(-Z_test[i,0],Z_test[i,1],color=point_colors[np.int(Y_test[i])],s=5)


# In[81]:


train_pred=[]

for i in range(len(Z_train[:,1])):
    ax=Z_train[i,:]
    in_dist=np.zeros(Z_train.shape[0])
    for j in range(len(Z_train[:,1])):
        in_dist[j]=dt.euclidean(ax,Z_train[j,:])
    least=np.argsort(in_dist)[:5]
    label=[]
    for m in least:
        label.append(Y_train[m])
    predict=stats.mode(label)[0]
    train_pred.append(predict)
print(confusion_matrix(train_pred,Y_train))


# In[83]:


test_pred = []

for i in range(len(Z_test[:,1])):
    ax = Z_test[i, :]
    in_dist = np.zeros(Z_train.shape[0])
    for j in range(len(Z_train[:,1])):
        in_dist[j] = dt.euclidean(ax, Z_train[j, :])
    least = np.argsort(in_dist)[:5]
    label=[]
    for m in least:
        label.append(Y_train[m])
    predict= stats.mode(label)[0]
    test_pred.append(predict)
print(confusion_matrix(train_pred,Y_train))


# In[ ]:




