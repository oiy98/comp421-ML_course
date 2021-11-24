#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
def safelog(x):
    return(np.log(x + 1e-100))


# In[47]:


# read data into memory
images = np.genfromtxt("hw03_data_set_images.csv",delimiter=",")

labels = pd.read_csv("hw03_data_set_labels.csv").values

training_set=[]
test_set=[]
for i in range(5):
    for j in range(25):
        training_set.append(images[39*i+j,:])
for i in range(5):
    for j in range(14):
        test_set.append(images[39*i+j+25,:])
training_set=np.array(training_set)
test_set=np.array(test_set)
y_truth=[] #labels as integers
for i in range(labels.shape[0]):
    if labels[i,0]=='A':
        y_truth.append(1)
    elif labels[i,0]=='B':
        y_truth.append(2)
    elif labels[i,0]=='C':
        y_truth.append(3)
    elif labels[i,0]=='D':
        y_truth.append(4)
    elif labels[i,0]=='E':
        y_truth.append(5)
y_truth.append(5)
y_truth=np.array(y_truth)
train_truth=[]
train_t_coded=[]
for k in range(1,6):
    for i in range(25):
        train_truth.append(k)
        if k==1:
            train_t_coded.append([1,0,0,0,0])
        if k==2:
            train_t_coded.append([0,1,0,0,0])
        if k==3:
            train_t_coded.append([0,0,1,0,0])
        if k==4:
            train_t_coded.append([0,0,0,1,0])
        if k==5:
            train_t_coded.append([0,0,0,0,1])
train_truth=np.array(train_truth)
train_t_coded=np.array(train_t_coded)

test_t_coded=[]
test_truth=[]

for k in range(1,6):
    for i in range(14):
        test_truth.append(k)
        if k==1:
            test_t_coded.append([1,0,0,0,0])
        if k==2:
            test_t_coded.append([0,1,0,0,0])
        if k==3:
            test_t_coded.append([0,0,1,0,0])
        if k==4:
            test_t_coded.append([0,0,0,1,0])
        if k==5:
            test_t_coded.append([0,0,0,0,1])
test_truth=np.array(test_truth)
test_t_coded=np.array(test_t_coded)

K=5
N= (125,320)


# In[48]:


#estimate parameter
PCD=np.zeros((5,320))
for i in range(5):
    for k in range(320):
        a=np.mean(training_set[(25*i):(25*i+25),k])
        PCD[i,k]=a
print(PCD)


# In[49]:


plt.imshow(PCD[0].reshape(16,20).T, cmap='hot', interpolation='spline16')
plt.show()
plt.imshow(PCD[1].reshape(16,20).T, cmap='hot', interpolation='spline16')
plt.show()
plt.imshow(PCD[2].reshape(16,20).T, cmap='hot', interpolation='spline16')
plt.show()
plt.imshow(PCD[3].reshape(16,20).T, cmap='hot', interpolation='spline16')
plt.show()
plt.imshow(PCD[4].reshape(16,20).T, cmap='hot', interpolation='spline16')
plt.show()


# In[50]:


#Calculate the confusion matrix for the data points in your training set using the
#parametric discrimination rule you will develop using the estimated parameters. 


# In[51]:


#class priors
#yani P(y=0,1,2,3,4)
priors =[1/5,1/5,1/5,1/5,1/5]


# In[52]:


def score(x):
    scores = np.zeros(5)
    for i in range(5):
        scores[i] = scores[i] + safelog(priors[i])
        scores[i] = scores[i] + np.sum( x*safelog(PCD[i]) + ( (np.ones(320) - x)*safelog(np.ones(320) - PCD[i])) )
    return scores


# In[53]:


def decide(a):
    scores=score(a)
    if(scores[0]==max(scores)):
        return 0
    elif(scores[1]==max(scores)):
        return 1
    elif(scores[2]==max(scores)):
        return 2
    elif(scores[3]==max(scores)):
        return 3
    else:
        return 4


# In[54]:


train_decided=[]
for i in range(training_set.shape[0]):
    a=decide(training_set[i])
    train_decided.append(a)
    
train_decided=np.array(train_decided)


# In[55]:


confusion_matrix = pd.crosstab(train_decided, train_truth, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)


# In[56]:


#for test set
#estimate parameter
PCDtest=np.zeros((5,320))
for i in range(5):
    for k in range(320):
        a=np.mean(test_set[(14*i):(14*i+14),k])
        PCDtest[i,k]=a
print(PCDtest)


# In[57]:


test_decided=[]
for i in range(test_set.shape[0]):
    a=decide(test_set[i])
    test_decided.append(a)
    
test_decided=np.array(test_decided)


# In[58]:


confusion_matrix = pd.crosstab(test_decided, test_truth, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)


# In[ ]:




