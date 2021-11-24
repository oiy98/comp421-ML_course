#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
def safelog(x):
    return(np.log(x + 1e-100))


# In[3]:


# read data into memory
data = np.genfromtxt("hw04_data_set.csv",delimiter=",")

train_set=[]
test_set=[]

for i in range(1,101):
        train_set.append(data[i,:])
for i in range(101,134):
        test_set.append(data[i,:])

train_set=np.array(train_set)
test_set=np.array(test_set)

x_train=train_set[:,0]
y_train=train_set[:,1]

x_test=test_set[:,0]
y_test=test_set[:,1]


N=data.shape[0]-1


# In[4]:


#regressogram
bin_width =3
origin=0


# In[5]:


point_colors = np.array(["red", "blue"])
minimum_value = 0
maximum_value = 60
data_interval = np.arange(minimum_value,maximum_value,step = 0.01)


# In[6]:


left_borders = np.arange(minimum_value,maximum_value-bin_width,step = bin_width)
right_borders = np.arange(minimum_value + bin_width,maximum_value,step = bin_width)


# In[7]:


p_hat = []
for x in range (len(left_borders)):
    p_hat.append((sum(((left_borders[x] < x_train) & (x_train <= right_borders[x]))*y_train)/(sum((left_borders[x] < x_train) &(x_train <= right_borders[x])))))
p_hat=np.array(p_hat)
plt.figure(figsize = (10, 6))
plt.plot(x_train,y_train,'ro')
plt.plot(x_test,y_test,'bo')
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [p_hat[b], p_hat[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [p_hat[b], p_hat[b + 1]], "k-")    
plt.show()


# In[8]:


#RMSE
rmse = 0
for i in range(0,len(x_test)):
    for b in range(0,len(left_borders)):
        if(left_borders[b] < x_test[i] and x_test[i] <= right_borders[b]):
            a = (y_test[i] - p_hat[int((x_test[i])/bin_width)])**2
            rmse= a+rmse
RMSE = np.sqrt(rmse / len(x_test))
print("Regressogram => RMSE is", RMSE," when h is 3")


# In[51]:


#running mean smoother
p_hat_rms = []
for x in range (len(data_interval)):
    k=sum((((data_interval[x]-0.5*bin_width)<x_train)&(x_train <= (data_interval[x] + 0.5 * bin_width)))*y_train)
    m=sum((((data_interval[x]-0.5*bin_width)<x_train)&(x_train <= (data_interval[x] + 0.5 * bin_width))))
    c = np.divide(k, m, out=np.zeros_like(k), where=m!=0)
    p_hat_rms.append(c)
p_hat_rms=np.array(p_hat_rms)

plt.figure(figsize = (10, 6))
plt.plot(x_train,y_train,'ro')
plt.plot(x_test,y_test,'bo')
plt.plot(data_interval,p_hat_rms)


# In[46]:


#RMSE
rmse_rms = 0
for i in range(len(x_test)):
    b = (y_test[i] - p_hat_rms[int((x_test[i])/0.01)])**2
    rmse_rms = b +rmse_rms
RMSE_rms = np.sqrt(rmse_rms / len(x_test))
print("Regressogram => RMSE is", RMSE_rms," when h is 3")


# In[13]:


#kernel smoother
a=[]
b=[]
bin_width=1
p_hat_kernel = []
for x in range(len(data_interval)):
    a= sum(((1 / math.sqrt(2 * math.pi)) * (np.exp(-0.5 * ((x_train-data_interval[x])**2) / bin_width**2))*y_train)) 
    b= sum((1 / math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((x_train-data_interval[x])**2) / bin_width**2))
    p_hat_kernel.append(a/b)
plt.figure(figsize = (10, 6))
plt.plot(x_train,y_train,'ro')
plt.plot(x_test,y_test,'bo')
plt.plot(data_interval,p_hat_kernel)


# In[24]:


#RMSE
rmse_kernel = 0
for i in range(len(x_test)):
    a = (y_test[i] - p_hat_kernel[int((x_test[i])/0.01)])**2
    rmse_kernel = a +rmse_kernel
RMSE_kernel = np.sqrt(rmse_kernel / len(x_test))
print("Regressogram => RMSE is", RMSE_kernel," when h is 1")

