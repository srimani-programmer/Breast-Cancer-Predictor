#!/usr/bin/env python
# coding: utf-8

# In[14]:


# import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[15]:


# import requests
# url = 'https://raw.githubusercontent.com/vanshu25/Breast-Cancer-Predictor/master/Breast%20Cancer%20Data.csv'
# res = requests.get(url, allow_redirects=True)
# with open('breast_cancer.csv','wb') as file:
#     file.write(res.content)
df = pd.read_csv('breast_cancer.csv')


# In[16]:


#Our main purpose is to classify whether it is Malignant=M or Benign=B..so let's convert M and into 1 and 0
df['diagnosis'].replace({'M':1,'B':0},inplace=True)


# In[17]:


#we cans ee that 'unnamed' is not contributing to the classification,so we will drop this column
X = df.drop(['Unnamed: 32','id','diagnosis'],axis=1)
Y = df['diagnosis'].to_numpy()


# In[18]:


#splitting data into training and test set
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.30)


# In[19]:


#let's perfrom standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[20]:


# create the model
model = torch.nn.Linear(X_train.shape[1], 1)


# In[21]:


# load sets in format compatible with pytorch
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))


# In[22]:


y_train = torch.from_numpy(y_train).float().reshape(-1, 1)
y_test = torch.from_numpy(y_test).float().reshape(-1, 1)


# In[23]:


#Let's specify hyperparameters and iterate through train data to run the model
def Loss_function():
    return torch.nn.BCEWithLogitsLoss()


# In[24]:


#for gradient descent
def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr = 0.0007)


# In[25]:


criterion = Loss_function()
Optimizer = optimizer(model)


# In[26]:


# run the model
epochs = 2000
# initialise the train_loss & test_losses which will be updated
train_losses = np.zeros(epochs)
test_losses = np.zeros(epochs)

for epoch in range(epochs): 
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    # clear old gradients from the last step
    Optimizer.zero_grad()
    # compute the gradients necessary to adjust the weights
    loss.backward()
     # update the weights of the neural network
    Optimizer.step()

    outputs_test = model(X_test)
    loss_test = criterion(outputs_test, y_test)

    train_losses[epoch] = loss.item()
    test_losses[epoch] = loss_test.item()

    if (epoch + 1) % 50 == 0:
        print (str('Epoch ') + str((epoch+1)) + str('/') + str(epochs) + str(',  training loss = ') + str((loss.item())) + str(', test loss = ') + str(loss_test.item()))


# In[27]:





# In[ ]:




