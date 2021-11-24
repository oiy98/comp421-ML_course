#!/usr/bin/env python
# coding: utf-8

# In[83]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def safelog2(x):
    if x == 0:
        return (0)
    else:
        return (np.log2(x))


# In[84]:


# read data into memory
data = np.genfromtxt("hw05_data_set.csv",delimiter=",")

train_set=[]
test_set=[]

X = data[:,0]
y = data[:,1].astype(int)

for i in range(1,101):
        train_set.append(data[i,:])
for i in range(101,134):
        test_set.append(data[i,:])

train_set=np.array(train_set)
test_set=np.array(test_set)

K = np.max(y)
N = X.shape[0]
D = 1

X_train=train_set[:,0]
y_train=train_set[:,1]

X_test=test_set[:,0]
y_test=test_set[:,1]


N=data.shape[0]-1
N_train= len(y_train)
N_test=len(y_test)

P=15


# In[85]:


def algorithm(P):
    # learning algorithm
    splits={}
    node_means={}
    node_indices = {}
    is_terminal = {}
    node_split = {}

    # put all training instances into the root node
    node_indices[1] = np.array(range(N_train))
    is_terminal[1] = False
    node_split[1] = True
    while True:
        # find nodes that need splitting
        split_nodes = [key for key, value in node_split.items() if value == True]
        # check whether we reach all terminal nodes
        if len(split_nodes) == 0:
            break
        # find best split positions for all nodes
        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            node_split[split_node] = False
            node_mean = np.mean(y_train[data_indices])

            if X_train[data_indices].size <= P:
                is_terminal[split_node] = True
                node_means[split_node]=node_mean
            else:
                is_terminal[split_node] = False
                special_values = np.sort(np.unique(X_train[data_indices]))
                split_positions = (special_values[1:len(special_values)] + special_values[0:(len(special_values) - 1)]) / 2
                split_scores = np.repeat(0.0, len(split_positions))
                for s in range(len(split_positions)):
                    left_indices = data_indices[X_train[data_indices] < split_positions[s]]
                    right_indices = data_indices[X_train[data_indices] >= split_positions[s]]
                    total_error=0
                    if len(left_indices>0):
                        mean=np.mean(y_train[left_indices])
                        total_error= total_error+np.sum((y_train[left_indices]-mean)**2)
                    if len(right_indices>0):
                        mean= np.mean(y_train[right_indices])
                        total_error=total_error+np.sum((y_train[right_indices]-mean)**2)
                    split_scores[s]=total_error/(len(left_indices)+len(right_indices))
                    if len(special_values)==1:
                        is_terminal[split_node]= True
                        node_means[split_node]=node_mean
                        continue
                    best_splits= split_positions[np.argmin(split_scores)]
                    splits[split_node]=best_splits

                    # create left node using the selected split
                    left_indices = data_indices[(X_train[data_indices] < best_splits)]
                    node_indices[2 * split_node] = left_indices
                    is_terminal[2 * split_node] = False
                    node_split[2 * split_node] = True

                    # create right node using the selected split
                    right_indices = data_indices[(X_train[data_indices] >= best_splits)]
                    node_indices[2 * split_node + 1] = right_indices
                    is_terminal[2 * split_node + 1] = False
                    node_split[2 * split_node + 1] = True
    result=dict()
    result['node_indices']=node_indices
    result['is_terminal']=is_terminal
    result['node_split']=node_split
    result['node_means']=node_means
    result['splits']=splits
    return result


# In[118]:


Result15=dict()
Result15= algorithm(15)


# In[119]:


def prediction(x,is_terminal,node_means,splits):
    index = 1
    while (1):
        if (is_terminal[index] == True):
            return(node_means[index])
        if (x > splits[index]):
            index = index * 2 + 1
        else :
            index = index * 2


# In[120]:


point_colors = np.array(["red", "blue"])
minimum_value = 0
maximum_value = 60
data_interval = np.arange(minimum_value,maximum_value,step = 0.01)


# In[121]:


p_hat=[]
for b in range(len(data_interval)):
    x_left = data_interval[b]
    p_hat.append(prediction(x_left,Result15['is_terminal'],Result15['node_means'],Result15['splits']))

plt.figure(figsize = (10, 6))
plt.plot(X_train,y_train,'ro')
plt.plot(X_test,y_test,'bo')
plt.plot(data_interval,p_hat)
plt.xlabel('x')
plt.ylabel('y')
plt.title('P=15')


# In[122]:


#RMSE For test data
y_test_predicted =[]
for i in range(N_test):
    y_test_predicted.append(prediction(X_test[i],Result15['is_terminal'],Result15['node_means'],Result15['splits']))

RMSE = np.sqrt(sum((y_test - y_test_predicted) ** 2) / len(y_test))
print("RMSE is",RMSE," when P is ",  P)


# In[91]:


Rmses=[]
for i in range(1,11):
    y_pred=[]
    Pnew=i*5
    alg=dict()
    alg=algorithm(Pnew)
    for i in range(N_test):
        y_pred.append(prediction(X_test[i],alg['is_terminal'],alg['node_means'],alg['splits']))
    Rmses.append(np.sqrt(sum((y_test - y_pred) ** 2) / len(y_test)))
    


# In[95]:


Ps=np.arange(5,55,step = 5)
plt.figure(figsize = (10, 6))
plt.plot(Ps,Rmses,marker='o')
plt.xlabel('Pre−pruning size (P)')
plt.ylabel('RMSE')
plt.title('P=5,...,P=50')


# In[ ]:




