# -*- coding: utf-8 -*-
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")

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
Reading the CSV files
"""

#SC15
path=r"SC_15.csv"
df_SC15 = pd.read_csv(path)
df_SC15.head()

#SC19
path=r"SC_19.csv"
df_SC19 = pd.read_csv(path)
df_SC19.head()

#SC20
path=r"SC_20.csv"
df_SC20 = pd.read_csv(path)
df_SC20.head()

#SC23
path=r"SC_23.csv"
df_SC23 = pd.read_csv(path)
df_SC23.head()

#SC24
path=r"SC_24.csv"
df_SC24 = pd.read_csv(path)
df_SC24.head()

#SC27
path=r"SC_27.csv"
df_SC27 = pd.read_csv(path)
df_SC27.head()

#%%

#Total areas
df_train = pd.concat([
    df_SC23,#3.5 GHz
    df_SC24, #2.54 GHz
    df_SC27, #735 MHz   
    ])

df_test = pd.concat([
    df_SC15, #735 MHz
    df_SC19, #3.5 GHz
    df_SC20 #2.54 GHz
   
    ])

#%%
"""
Create Input and Output Data
"""

X_train = df_train.iloc[:, [6,15,11,17,12,13,7,14,10,8,9,16]] 
y_train = df_train.iloc[:, [25]]

X_test = df_test.iloc[:, [6,15,11,17,12,13,7,14,10,8,9,16]] 
y_test = df_test.iloc[:, [25]]

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

svr_regressor = SVR(kernel='rbf', C=100, epsilon=0.01,gamma=0.02)
svr_regressor.fit(X_train,np.ravel(y_train))

#TRAINING
y_pred = svr_regressor.predict(X_train)

y_pred = y_pred.reshape(-1,1)

y_pred_desn = scaler.inverse_transform(y_pred)
y_target_desn = scaler.inverse_transform(y_train)

pl_pred=y_pred_desn

df_train['pl_pred']=pl_pred

df_train_735=df_train.loc[df_train['freq'] == 735]
df_train_2540=df_train.loc[df_train['freq'] == 2540]
df_train_3500=df_train.loc[df_train['freq'] == 3500]

MSE = np.square(np.subtract(df_train_735['PL'],df_train_735['pl_pred'])).mean()
RMSE_735 = math.sqrt(MSE)

R2_735= r2_score(df_train_735['PL'],df_train_735['pl_pred']) #R2

MSE = np.square(np.subtract(df_train_2540['PL'],df_train_2540['pl_pred'])).mean()
RMSE_2540 = math.sqrt(MSE)

R2_2540= r2_score(df_train_2540['PL'],df_train_2540['pl_pred']) #R2

MSE = np.square(np.subtract(df_train_3500['PL'],df_train_3500['pl_pred'])).mean()
RMSE_3500 = math.sqrt(MSE)

R2_3500= r2_score(df_train_3500['PL'],df_train_3500['pl_pred']) #R2

#variance
variance_735 = np.sum((df_train_735['PL'] - df_train_735['PL'].mean())**2)/len(df_train_735['PL'])
variance_2540 = np.sum((df_train_2540['PL'] - df_train_2540['PL'].mean())**2)/len(df_train_2540['PL'])
variance_3500 = np.sum((df_train_3500['PL'] - df_train_3500['PL'].mean())**2)/len(df_train_3500['PL'])

#mean
mean_735=np.mean(df_train_735['PL'])
mean_2540=np.mean(df_train_2540['PL'])
mean_3500=np.mean(df_train_3500['PL'])

#Training
data = [['735',RMSE_735, R2_735,variance_735,mean_735],['2450',RMSE_2540, R2_2540,variance_2540,mean_2540]
        ,['3500',RMSE_3500, R2_3500,variance_3500,mean_3500]
        ]  
print(tabulate(data, headers=["Freq",'RMSE','R^2','Variance [dB]','Mean [dB]']))


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
y_pred_test = svr_regressor.predict(X_test)
y_pred_test = y_pred_test.reshape(-1,1)

y_pred_desn_test = scaler.inverse_transform(y_pred_test)

y_target_desn_test = scaler.inverse_transform(y_test)

pl_pred_test=y_pred_desn_test

df_test['pl_pred']=pl_pred_test

df_test_735=df_test.loc[df_test['freq'] == 735]
df_test_2540=df_test.loc[df_test['freq'] == 2540]
df_test_3500=df_test.loc[df_test['freq'] == 3500]

MSE = np.square(np.subtract(df_test_735['PL'],df_test_735['pl_pred'])).mean()
RMSE_735 = math.sqrt(MSE)

R2_735= r2_score(df_test_735['PL'],df_test_735['pl_pred']) #R2
R2_oos_735 = 1 - np.sum((df_test_735['PL'] - df_test_735['pl_pred'])**2) / np.sum((df_train_735['PL'].mean() - df_test_735['PL'])**2)

MSE = np.square(np.subtract(df_test_2540['PL'],df_test_2540['pl_pred'])).mean()
RMSE_2540 = math.sqrt(MSE)

R2_2540= r2_score(df_test_2540['PL'],df_test_2540['pl_pred']) #R2
R2_oos_2540 = 1 - np.sum((df_test_2540['PL'] - df_test_2540['pl_pred'])**2) / np.sum((df_train_2540['PL'].mean() - df_test_2540['PL'])**2)

MSE = np.square(np.subtract(df_test_3500['PL'],df_test_3500['pl_pred'])).mean()
RMSE_3500 = math.sqrt(MSE)

R2_3500= r2_score(df_test_3500['PL'],df_test_3500['pl_pred']) #R2
R2_oos_3500 = 1 - np.sum((df_test_3500['PL'] - df_test_3500['pl_pred'])**2) / np.sum((df_train_3500['PL'].mean() - df_test_3500['PL'])**2)

#Testing

#variance
variance_735 = np.sum((df_test_735['PL'] - df_test_735['PL'].mean())**2)/len(df_test_735['PL'])
variance_2540 = np.sum((df_test_2540['PL'] - df_test_2540['PL'].mean())**2)/len(df_test_2540['PL'])
variance_3500 = np.sum((df_test_3500['PL'] - df_test_3500['PL'].mean())**2)/len(df_test_3500['PL'])

#mean
mean_735=np.mean(df_test_735['PL'])
mean_2540=np.mean(df_test_2540['PL'])
mean_3500=np.mean(df_test_3500['PL'])

data = [['735',RMSE_735, R2_735,R2_oos_735,variance_735,mean_735],['2450',RMSE_2540, R2_2540,R2_oos_2540,variance_2540,mean_2540]
        ,['3500',RMSE_3500, R2_3500,R2_oos_3500,variance_3500,mean_3500]
        ]  
print(tabulate(data, headers=["Freq",'RMSE','R^2','R^2 OOS','Variance [dB]','Mean[dB]']))


#Plot
p1 = max(max(y_target_desn_test), max(y_target_desn_test))
p2 = min(min(y_target_desn_test), min(y_target_desn_test))
plt.plot([p1, p2], [p1, p2], '--',linewidth=1,label= 'Best fit',c='navy')

plt.scatter(y_target_desn_test,y_pred_desn_test,label='Result for Route #2 ($R^2:0.86$)')
plt.grid(ls='--')
plt.legend()
plt.xlabel('Measured path-loss [dB]')
plt.ylabel('Predicted path-loss [dB]')
plt.savefig('R2_V2I_BW.eps',format='eps',dpi=1200)
plt.show()
plt.close()

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

epsilon = [0.0001,0.001,0.01] 
    
gamma = [0.0002,0.002,0.02]  

max_iter = len(epsilon)*len(gamma)  

iteration = list(range(1,max_iter+1))

C = np.arange(0, 105, 5) 

p = len(C)*max_iter
zmatrix_svr = np.zeros((p, 5)) #size of the cross-validation matriz

cv = KFold(n_splits=5,shuffle=True,random_state=0)

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

#iterations
for i in range(max_iter):
    
    for k in range(len(C)): 
        
        model = SVR(kernel='rbf', C=C[k], epsilon=epsilon[md],gamma=gamma[ml])
        

        RMSE_tot_train = 0
        RMSE_tot_val = 0
        
        for train, test in cv.split(X_train, y_train):
            model.fit(X_train[train], np.ravel(y_train[train]))
            y_pred_train = model.predict(X_train[train])
            y_pred_val = model.predict(X_train[test])
            
            #desnormalization val
            y_pred_val = y_pred_val.reshape(-1,1)
            y_pred_desn_val = scaler.inverse_transform(y_pred_val)
            y_target_desn_val = scaler.inverse_transform(y_train[test])
            
            #desnormalization train
            y_pred_train = y_pred_train.reshape(-1,1)
            y_pred_desn_train = scaler.inverse_transform(y_pred_train)
            y_target_desn_train = scaler.inverse_transform(y_train[train])
            
            #RMSE training
            MSE = np.square(np.subtract(y_target_desn_train,y_pred_desn_train)).mean()
            RMSE_desn_train = math.sqrt(MSE)
            RMSE_tot_train +=RMSE_desn_train
            
            #RMSE testing
            MSE = np.square(np.subtract(y_target_desn_val,y_pred_desn_val)).mean()
            RMSE_desn_val = math.sqrt(MSE)
            RMSE_tot_val +=RMSE_desn_val
            
        RMSE_tot_train = RMSE_tot_train/5
        RMSE_tot_val = RMSE_tot_val/5
        
        #save grid search results
        zmatrix_svr[count][0] = C[k]
        zmatrix_svr[count][1] = epsilon[md]
        zmatrix_svr[count][2] = gamma[ml]
        zmatrix_svr[count][3] = RMSE_tot_train
        zmatrix_svr[count][4] = RMSE_tot_val
        count += 1
        c +=1
        
        if c == a + len(C)+1:
            md +=1
            a = a + len(C)
            x +=1
            
            
        if x == len(epsilon):
            x=0
            md=0
        
        if c == b + len(C)*len(epsilon)+1:
            ml +=1
            b = b + len(C)*len(epsilon)




