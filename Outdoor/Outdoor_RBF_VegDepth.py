# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:24:22 2023
@author: Yoiz Nuñez
"""

#1 Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data import SubsetRandomSampler #split the dataset

from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import math
from sklearn.neural_network import MLPRegressor
from sklearn import gaussian_process
from sklearn.gaussian_process import kernels

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

torch.manual_seed(0)
np.random.seed(0)

import random
random.seed(0)

#%%
"""
Read Data
"""
path = r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\Outdoor_data_vegetationdepth.csv"
df = pd.read_csv(path)
df.head()

"""
Create Input and Output Data
"""
X = df.iloc[:, [0,1,2,5]]
y = df.iloc[:, [3]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#normalize inputs
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

#normalize output
y_train = y_train.reshape(-1, 1)
y_train = scaler.fit_transform(y_train)
y_test = y_test.reshape(-1, 1)
y_test = scaler.transform(y_test)

# convert output variable to float
y_train, y_test = y_train.astype(float), y_test.astype(float),

#%%

l=7.2
kernel=kernels.RBF(length_scale=l)

#%%

gp=gaussian_process.GaussianProcessRegressor(kernel=kernel,optimizer=None,random_state=42)

gp.fit(X_train, y_train)

#%%
#kernel = 3 * RBF(length_scale=1e-1,length_scale_bounds=(1e-3,1e3))
#gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
#gaussian_process.fit(X_train, y_train)

#%%
#Train
y_pred = gp.predict(X_train)

y_pred = y_pred.reshape(-1,1)

y_pred_desn = scaler.inverse_transform(y_pred)
y_target_desn = scaler.inverse_transform(y_train)

MSE = np.square(np.subtract(y_target_desn,y_pred_desn)).mean()
RMSE_train = math.sqrt(MSE)
print(RMSE_train)


#Test
y_pred = gp.predict(X_test)

y_pred = y_pred.reshape(-1,1)

y_pred_desn = scaler.inverse_transform(y_pred)
y_target_desn = scaler.inverse_transform(y_test)

MSE = np.square(np.subtract(y_target_desn,y_pred_desn)).mean()
RMSE_test = math.sqrt(MSE)
print(RMSE_test)

#%%

"""
Applying Cross-Validation
"""
#learning_rate = [0.001,0.01,0.1] #define maximum depth

#alpha = [0.001,0.01,0.1] 

#max_iter = len(learning_rate)*len(alpha)

#iteration = list(range(1,max_iter+1))

length = np.arange(1,10,0.1) # Number of neurons

p = len(length)
matrix_cv_rbf = np.zeros((p, 2)) # Size of the CV matriz

cv = KFold(n_splits=5,shuffle=True,random_state=42)

#%%
count = 0
md = 0
ml = 0
lr = 0
c = 1
a = 0
b = 0
x=0
bx=0



for k in range(len(length)):
    
    l=np.sqrt(length[k])
    kernel=kernels.RBF(length_scale=l)

    model=gaussian_process.GaussianProcessRegressor(kernel=kernel,optimizer=None,random_state=42)
    
    RMSE_tot = 0
    
    for train, test in cv.split(X_train, y_train):
        model.fit(X_train[train], np.ravel(y_train[train]))
        y_pred = model.predict(X_train[test])
        
        #desnormalization
        y_pred = y_pred.reshape(-1,1)
        y_pred_desn = scaler.inverse_transform(y_pred)
        y_target_desn = scaler.inverse_transform(y_train[test])
        
        #RMSE
        MSE = np.square(np.subtract(y_target_desn,y_pred_desn)).mean()
        RMSE_desn = math.sqrt(MSE)
        RMSE_tot +=RMSE_desn
        
    RMSE_tot = RMSE_tot/5
   
    matrix_cv_rbf[count][0] = length[k]
    matrix_cv_rbf[count][1] = RMSE_tot
    count += 1
    
