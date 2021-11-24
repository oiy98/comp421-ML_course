
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def safelog(x):
    return(np.log(x + 1e-100))


# In[2]:



# read data into memory
images = np.genfromtxt("hw02_data_set_images.csv",delimiter=",")

labels = pd.read_csv("hw02_data_set_labels.csv").as_matrix()

training_set=[]
test_set=[]
for i in range(5):
    for j in range(25):
        training_set.append(images[39*i+j,:])
for i in range(5):
    for j in range(14):
        test_set.append(images[39*i+j+14,:])
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
N=images.shape[0]
K= max(y_truth)
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


# In[3]:


# define the sigmoid function
def sigmoid(X, w, w0):
    return(1 / (1 + np.exp(-(np.matmul(w.T,X.T) + w0.T))))


# In[4]:


eta= 0.01
epsilon=1e-3


# In[5]:


# randomly initalize W and w0
np.random.seed(421)
W = np.random.uniform(low = -0.01, high = 0.01, size = (training_set.shape[1], K))
w0 = np.random.uniform(low = -0.01, high = 0.01, size = (1, 5))


# In[6]:


# define the gradient functions
def gradient_W(X, Y_truth, Y_predicted):
    Y_predicted=Y_predicted.T
    return(np.asarray([-np.sum(np.repeat((Y_truth[:,c] - Y_predicted[:,c])[:, None], X.shape[1], axis = 1) * X, axis = 0) for c in range(5)]).transpose())

def gradient_w0(Y_truth, Y_predicted):
    return(-np.sum(Y_truth - Y_predicted, axis = 0))


# In[7]:


# learn W and w0 using gradient descent
iteration = 0
objective_values = []
while 1:
    Y_predicted = sigmoid(training_set, W, w0)

    objective_values = np.append(objective_values, 1/2*np.sum((train_t_coded-Y_predicted.T)**2))
    W_old = W
    w0_old = w0

    W = W - eta * gradient_W(training_set, train_t_coded, Y_predicted)
    w0 = w0 - eta * gradient_w0(train_t_coded, Y_predicted.T)

    if np.sqrt(np.sum((w0 - w0_old))**2 + np.sum((W - W_old)**2)) < epsilon:
        break

    iteration = iteration + 1
print(W)
print(w0)


# In[8]:


# plot objective function during iterations
plt.figure(figsize = (10, 6))
plt.plot(range(1, iteration + 2), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


# In[9]:


Y_predicted_coded=[]
for i in range(Y_predicted.shape[1]):
    if Y_predicted[0,i]==max(Y_predicted[:,i]):
        Y_predicted_coded.append(0)
    if Y_predicted[1,i]==max(Y_predicted[:,i]):
        Y_predicted_coded.append(1)
    if Y_predicted[2,i]==max(Y_predicted[:,i]):
        Y_predicted_coded.append(2)
    if Y_predicted[3,i]==max(Y_predicted[:,i]):
        Y_predicted_coded.append(3)
    if Y_predicted[4,i]==max(Y_predicted[:,i]):
        Y_predicted_coded.append(4)
Y_predicted_coded=np.array(Y_predicted_coded)
confusion_matrix = pd.crosstab(Y_predicted_coded, train_truth, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)


# In[10]:


Y_pred=sigmoid(test_set,W,w0)
Y_pred_coded=[]
for i in range(Y_pred.shape[1]):
    if Y_pred[0,i]==max(Y_pred[:,i]):
        Y_pred_coded.append(0)
    if Y_pred[1,i]==max(Y_pred[:,i]):
        Y_pred_coded.append(1)
    if Y_pred[2,i]==max(Y_pred[:,i]):
        Y_pred_coded.append(2)
    if Y_pred[3,i]==max(Y_pred[:,i]):
        Y_pred_coded.append(3)
    if Y_pred[4,i]==max(Y_pred[:,i]):
        Y_pred_coded.append(4)
Y_pred_coded=np.array(Y_pred_coded)
confusion_matrix = pd.crosstab(Y_pred_coded, test_truth, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)

