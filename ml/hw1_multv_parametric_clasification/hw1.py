
# coding: utf-8

# In[2]:


import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp
import pandas as pd


# In[3]:


np.random.seed(421)
# mean parameters
class_means = np.array([[+0.0, +2.5],[-2.5,-2.0],[+2.5,-2.0]])
# standard deviation parameters
class_covariances = np.array([[[+3.2,+0.0],
                             [+0.0,+1.2]],
                             [[+1.2,-0.8],
                              [-0.8,+1.2]],
                             [[+1.2,+0.8],
                              [+0.8,+1.2]]])
# sample sizes
class_sizes = np.array([120, 90, 90])


# In[4]:


# generate random samples
points1 = np.random.multivariate_normal(class_means[0,:], class_covariances[0,:,:], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1,:], class_covariances[1,:,:], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2,:], class_covariances [2,:,:],class_sizes[2])
X = np.vstack((points1, points2 ,points3))

# generate corresponding labels
y1 = np.concatenate((np.repeat(0, class_sizes[0]), np.repeat(1, class_sizes[1])))
y  = np.concatenate((y1,np.repeat(2,class_sizes[2])))


# In[5]:


# plot data points generated
plt.figure(figsize = (10, 10))
plt.plot(points1[:,0], points1[:,1], "r.", markersize = 10)
plt.plot(points2[:,0], points2[:,1], "g.", markersize = 10)
plt.plot(points3[:,0], points3[:,1], "b.", markersize=10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# In[6]:


#number of labels and number of samples
K= max(y)
L=len(y)
print(K,L)


# In[7]:


x1_mean=(np.mean(points1[:,0]),np.mean(points1[:,1]))
x2_mean=(np.mean(points2[:,0]),np.mean(points2[:,1]))
x3_mean=(np.mean(points3[:,0]),np.mean(points3[:,1]))
samp_mean=[]
samp_mean.append(x1_mean)
samp_mean.append(x2_mean)
samp_mean.append(x3_mean)
print(samp_mean)


# In[8]:


#sample covariance
def samp_cov(x,index):
    A=np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            for k in range(class_sizes[index]):
                A[i,j]=A[i,j]+(x[k,i]-np.mean(x[:,i]))*(x[k,j]-np.mean(x[:,j]))
    return(A/class_sizes[index])
cov1=samp_cov(points1,0)
cov2=samp_cov(points2,1)
cov3=samp_cov(points3,2)
samp_covs=[]
samp_covs.append(cov1)
samp_covs.append(cov2)
samp_covs.append(cov3)
print(samp_covs)


# In[9]:


#class priors
priors=[]
size=class_sizes[0]+class_sizes[1]+class_sizes[2]
for i in range(3):
    priors.append((class_sizes[i])/size)
print(priors)


# In[10]:


y_predicted = []
for i in range(X.shape[0]):
    scores = [ -1/2 * np.dot(np.dot(X[i] - samp_mean[c], np.linalg.inv(samp_covs[c])), X[i] - samp_mean[c]) + np.log(priors[c]) for c in range(3)]
    y_predicted.append(np.argmax(scores))
print(y_predicted)
y_predicted = np.array(y_predicted)
print(y_predicted)


# In[151]:


# calculate confusion matrix
confusion_matrix = pd.crosstab(y_predicted, y, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)


# In[212]:


def predict(X):
    y_predicted=[]
    for c in range(3):
        scores = -1/2 * np.dot(np.dot(X - samp_mean[c], np.linalg.inv(samp_covs[c])), X - samp_mean[c]) + np.log(priors[c])
        y_predicted.append(scores)
    return y_predicted
# evaluate discriminant function on a grid
x1_interval=np.linspace(-5,5,1000)
x2_interval=np.linspace(-5,5,1000)
discriminant_values = np.zeros((len(x1_interval), len(x2_interval), K+1))

point=np.zeros((1,2))
for i in range(x1_interval.shape[0]):
    for j in range(x2_interval.shape[0]):
        discriminant_values[i][j][:]=predict(np.array(x1_interval[i],x2_interval[j]))


# In[213]:


A = discriminant_values[:,:,0]
B = discriminant_values[:,:,1]
C = discriminant_values[:,:,2]
A[(A < B) & (A < C)] = np.nan
B[(B < A) & (B < C)] = np.nan
C[(C < A) & (C < B)] = np.nan
discriminant_values[:,:,0] = A
discriminant_values[:,:,1] = B
discriminant_values[:,:,2] = C

plt.figure(figsize = (10, 10))
plt.plot(X[y == 0, 0], X[y == 0, 1], "r.", markersize = 10)
plt.plot(X[y == 1, 0], X[y == 1, 1], "g.", markersize = 10)
plt.plot(X[y == 2, 0], X[y == 2, 1], "b.", markersize = 10)
plt.plot(X[y_predicted != y, 0], X[y_predicted != y, 1], "ko", markersize = 12, fillstyle = "none")
plt.contour(x1_interval, x2_interval, discriminant_values[:,:,0] - discriminant_values[:,:,1], levels=0,colors = "k")
plt.contour(x1_interval, x2_interval, discriminant_values[:,:,0] - discriminant_values[:,:,2], levels = 0, colors = "k")
plt.contour(x1_interval, x2_interval,discriminant_values[:,:,1] - discriminant_values[:,:,2], levels = 0, colors = "k")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

