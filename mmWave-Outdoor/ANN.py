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

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import numpy
import matplotlib.pyplot as plt

import tabulate
from tabulate import tabulate

torch.manual_seed(0)
np.random.seed(0)

import random
random.seed(0)

#%%
"""
Read Data
"""
path = r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\Outdoor_data_vegetationdepth_qgis_final.csv"
df = pd.read_csv(path)
df.head()

"""
Create Input and Output Data
"""
X = df.iloc[:, [0,1,2,5]]
y = df.iloc[:, [3]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=90)


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

mlp_regressor = MLPRegressor(hidden_layer_sizes=78,
                              activation='relu',
                              solver='lbfgs',
                              alpha=0.1,
                              learning_rate_init=0.1,
                              momentum=0.9,
                              random_state=42, 
                              max_iter=10000,
                              early_stopping=True)
  
mlp_regressor.fit(X_train,np.ravel(y_train))


#TRAINING
y_pred = mlp_regressor.predict(X_train)

y_pred = y_pred.reshape(-1,1)

y_pred_desn = scaler.inverse_transform(y_pred)
y_target_desn = scaler.inverse_transform(y_train)


MSE = np.square(np.subtract(y_target_desn,y_pred_desn)).mean() #RMSE
RMSE_train = math.sqrt(MSE)

MAPE_train = np.mean(np.abs((y_target_desn - y_pred_desn)/y_target_desn))*100 #MAPE

R2_train= r2_score(y_target_desn,y_pred_desn) #R2


n = len(y_pred_desn)
sum_model=0

abs_dif = np.abs(y_target_desn-y_pred_desn)
mean_model = np.mean(abs_dif)

for x in abs_dif:
    t = (x - mean_model) ** 2
    sum_model += t 

SD_train = math.sqrt(sum_model/(n)) #SD


#TESTING
y_pred_test = mlp_regressor.predict(X_test)
y_pred_test = y_pred_test.reshape(-1,1)

y_pred_desn_test = scaler.inverse_transform(y_pred_test)
y_target_desn_test = scaler.inverse_transform(y_test)


#Plot
p1 = max(max(y_target_desn_test), max(y_target_desn_test))
p2 = min(min(y_target_desn_test), min(y_target_desn_test))
plt.plot([p1, p2], [p1, p2], '--',linewidth=1,label= 'Best fit',c='navy')

plt.scatter(y_target_desn_test,y_pred_desn_test)
plt.show()

MSE = np.square(np.subtract(y_target_desn_test,y_pred_desn_test)).mean()
RMSE_test = math.sqrt(MSE)


R2_test= r2_score(y_target_desn_test,y_pred_desn_test) #R2


MAPE_test = np.mean(np.abs((y_target_desn_test - y_pred_desn_test)/y_target_desn_test))*100 #MAPE


n = len(y_pred_desn_test)
sum_model=0

abs_dif = np.abs(y_target_desn_test-y_pred_desn_test)
mean_model = np.mean(abs_dif)

for x in abs_dif:
    t = (x - mean_model) ** 2
    sum_model += t 

SD_test = math.sqrt(sum_model/(n)) #SD


#Training
data = [['Training set',RMSE_train, MAPE_train, SD_train,R2_train]]  
print(tabulate(data, headers=["","RMSE ",'MAPE [%]','SD','R^2']))

#Testing
data = [['Testing set',RMSE_test, MAPE_test, SD_test,R2_test]]
print(tabulate(data, headers=["","",'','','']))

#%%

"""
Applying Cross-Validation
"""
learning_rate = [0.001,0.01,0.1] #define maximum depth

alpha = [0.001,0.01,0.1] 

max_iter = len(learning_rate)*len(alpha)

iteration = list(range(1,max_iter+1))

num_neurons = np.arange(10,90,4) # Number of neurons

p = len(num_neurons)*max_iter
matrix_cv_mlp = np.zeros((p, 4)) # Size of the CV matriz

cv = KFold(n_splits=5,shuffle=True,random_state=42)


count = 0
md = 0
ml = 0
lr = 0
c = 1
a = 0
b = 0
x=0
bx=0

for i in range(max_iter):

    for k in range(len(num_neurons)):
        
        model = MLPRegressor(hidden_layer_sizes=num_neurons[k], 
                             activation='relu',
                             solver='lbfgs',
                             alpha=alpha[md],
                             learning_rate_init=learning_rate[ml],
                             momentum=0.9,
                             random_state=42, 
                             max_iter=15000,
                             early_stopping=True)
        
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
       
        matrix_cv_mlp[count][0] = num_neurons[k]
        matrix_cv_mlp[count][1] = learning_rate[md]
        matrix_cv_mlp[count][2] = alpha[ml]
        matrix_cv_mlp[count][3] = RMSE_tot
        count += 1
        c +=1
        
        if c == a + len(num_neurons)+1:
            md +=1
            a = a + len(num_neurons)
            x +=1
            
        if x == 3:
            x=0
            md=0
        
        if c == b + len(num_neurons)*len(learning_rate)+1:
            ml +=1
            b = b + len(num_neurons)*len(learning_rate)

#%%

#Campus Map of the University...

