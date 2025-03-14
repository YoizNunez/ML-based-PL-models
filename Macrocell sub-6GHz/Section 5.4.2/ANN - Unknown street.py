# -*- coding: utf-8 -*-

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

path=r"SC1_750.csv"
df_SC1_750 = pd.read_csv(path)
df_SC1_750.head()

path=r"SC1_2500.csv"
df_SC1_2500 = pd.read_csv(path)
df_SC1_2500.head()

path=r"SC1_3500.csv"
df_SC1_3500 = pd.read_csv(path)
df_SC1_3500.head()

path=r"SC2_750.csv"
df_SC2_750 = pd.read_csv(path)
df_SC2_750.head()

path=r"SC2_2500.csv"
df_SC2_2500 = pd.read_csv(path)
df_SC2_2500.head()

path=r"SC2_3500.csv"
df_SC2_3500 = pd.read_csv(path)
df_SC2_3500.head()

#%%
#Selection of samples for training and testing

#SC1
samples_test = 1700 #to take 700 sample on the testing set
samples_train_SC1_750 = len(df_SC1_750)  - samples_test
samples_train_SC1_2500 = len(df_SC1_2500) - samples_test
samples_train_SC1_3500 = len(df_SC1_3500) - 1470

#SC2
samples_test = 960 #to take the last 700 sample on the testing set
samples_train_SC2_750 = len(df_SC2_750)  - samples_test
samples_train_SC2_2500 = len(df_SC2_2500) - samples_test
samples_train_SC2_3500 = len(df_SC2_3500) - 880

#SC1
df_train_SC1_750=df_SC1_750.loc[np.r_[0:samples_train_SC1_750, samples_train_SC1_750+700:len(df_SC1_750)], :]
df_train_SC1_2500=df_SC1_2500.loc[np.r_[0:samples_train_SC1_2500, samples_train_SC1_2500+700:len(df_SC1_2500)], :]
df_train_SC1_3500=df_SC1_3500.loc[np.r_[0:samples_train_SC1_3500, samples_train_SC1_3500+545:len(df_SC1_3500)], :]

df_test_SC1_750= df_SC1_750.iloc[samples_train_SC1_750:samples_train_SC1_750+700]
df_test_SC1_2500= df_SC1_2500.iloc[samples_train_SC1_2500:samples_train_SC1_2500+700]
df_test_SC1_3500= df_SC1_3500.iloc[samples_train_SC1_3500:samples_train_SC1_3500+545]

#SC2
df_train_SC2_750= df_SC2_750.loc[np.r_[0:samples_train_SC2_750, samples_train_SC2_750+290:len(df_SC2_750)], :]
df_train_SC2_2500= df_SC2_2500.loc[np.r_[0:samples_train_SC2_2500, samples_train_SC2_2500+290:len(df_SC2_2500)], :]
df_train_SC2_3500= df_SC2_3500.loc[np.r_[0:samples_train_SC2_3500, samples_train_SC2_3500+275:len(df_SC2_3500)], :]

df_test_SC2_750= df_SC2_750.iloc[samples_train_SC2_750:samples_train_SC2_750+290]
df_test_SC2_2500= df_SC2_2500.iloc[samples_train_SC2_2500:samples_train_SC2_2500+290]
df_test_SC2_3500= df_SC2_3500.iloc[samples_train_SC2_3500:samples_train_SC2_3500+275]

#joining the multiples dataframe, to generate the final training and testing sets
df_train = pd.concat([df_train_SC1_750,df_train_SC1_2500,df_train_SC1_3500,df_train_SC2_750,df_train_SC2_2500,df_train_SC2_3500])
df_test_SC1 = pd.concat([df_test_SC1_750,df_test_SC1_2500,df_test_SC1_3500])
df_test_SC2 = pd.concat([df_test_SC2_750,df_test_SC2_2500,df_test_SC2_3500])

df_test = pd.concat([df_test_SC1,df_test_SC2])

fig, ax = plt.subplots(figsize = (8,7))
ax.scatter(df_train_SC1_750['Long'], df_train_SC1_750['Lat'], c='black', s=20)
ax.scatter(df_test_SC1_750['Long'], df_test_SC1_750['Lat'], c='blue', s=20)

fig, ax = plt.subplots(figsize = (8,7))
ax.scatter(df_train_SC1_2500['Long'], df_train_SC1_2500['Lat'], c='black', s=20)
ax.scatter(df_test_SC1_2500['Long'], df_test_SC1_2500['Lat'], c='green', s=20)

fig, ax = plt.subplots(figsize = (8,7))
ax.scatter(df_train_SC1_3500['Long'], df_train_SC1_3500['Lat'], c='black', s=20)
ax.scatter(df_test_SC1_3500['Long'], df_test_SC1_3500['Lat'], c='orange', s=20)

                  
#%%
"""
Create Input and Output Data
"""

X_train = df_train.iloc[:, [12,5,7,10,4,15,9,8,18,20,19,16,17,13]] #Predictors
y_train = df_train.iloc[:, [21]] #PL

X_test = df_test.iloc[:, [12,5,7,10,4,15,9,8,18,20,19,16,17,13]] 
y_test = df_test.iloc[:, [21]]


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

mlp_regressor = MLPRegressor(hidden_layer_sizes=19, 
                                 activation='relu',
                                 solver='lbfgs',
                                 alpha=0.1,
                                 learning_rate_init=0.001,
                                 momentum=0.9,
                                 random_state=0, 
                                 max_iter=6000,
                                 early_stopping=True)    
    
mlp_regressor.fit(X_train,np.ravel(y_train))    


#TRAINING
y_pred = mlp_regressor.predict(X_train)

y_pred = y_pred.reshape(-1,1)

y_pred_desn = scaler.inverse_transform(y_pred)
y_target_desn = scaler.inverse_transform(y_train)

pl_pred=y_pred_desn

df_train['pl_pred']=pl_pred

df_train_735=df_train.loc[df_train['freq'] == 750]
df_train_2540=df_train.loc[df_train['freq'] == 2500]
df_train_3500=df_train.loc[df_train['freq'] == 3500]

MSE = np.square(np.subtract(df_train_735['pl'],df_train_735['pl_pred'])).mean()
RMSE_735 = math.sqrt(MSE)

R2_735= r2_score(df_train_735['pl'],df_train_735['pl_pred']) #R2

MSE = np.square(np.subtract(df_train_2540['pl'],df_train_2540['pl_pred'])).mean()
RMSE_2540 = math.sqrt(MSE)

R2_2540= r2_score(df_train_2540['pl'],df_train_2540['pl_pred']) #R2

MSE = np.square(np.subtract(df_train_3500['pl'],df_train_3500['pl_pred'])).mean()
RMSE_3500 = math.sqrt(MSE)

R2_3500= r2_score(df_train_3500['pl'],df_train_3500['pl_pred']) #R2

#variance
variance_735 = np.sum((df_train_735['pl'] - df_train_735['pl'].mean())**2)/len(df_train_735['pl'])
variance_2540 = np.sum((df_train_2540['pl'] - df_train_2540['pl'].mean())**2)/len(df_train_2540['pl'])
variance_3500 = np.sum((df_train_3500['pl'] - df_train_3500['pl'].mean())**2)/len(df_train_3500['pl'])

#mean
mean_735=np.mean(df_train_735['pl'])
mean_2540=np.mean(df_train_2540['pl'])
mean_3500=np.mean(df_train_3500['pl'])

#Training
data = [['735',RMSE_735, R2_735,variance_735,mean_735],['2450',RMSE_2540, R2_2540,variance_2540,mean_2540],['3500',RMSE_3500, R2_3500,variance_3500,mean_3500]]  
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
y_pred_test = mlp_regressor.predict(X_test)
y_pred_test = y_pred_test.reshape(-1,1)

y_pred_desn_test = scaler.inverse_transform(y_pred_test)

y_target_desn_test = scaler.inverse_transform(y_test)


pl_pred_test=y_pred_desn_test

df_test['pl_pred']=pl_pred_test

df_test_735=df_test.loc[df_test['freq'] == 750]
df_test_2540=df_test.loc[df_test['freq'] == 2500]
df_test_3500=df_test.loc[df_test['freq'] == 3500]

MSE = np.square(np.subtract(df_test_735['pl'],df_test_735['pl_pred'])).mean()
RMSE_735 = math.sqrt(MSE)

R2_735= r2_score(df_test_735['pl'],df_test_735['pl_pred']) #R2
R2_oos_735 = 1 - np.sum((df_test_735['pl'] - df_test_735['pl_pred'])**2) / np.sum((df_train_735['pl'].mean() - df_test_735['pl'])**2)

MSE = np.square(np.subtract(df_test_2540['pl'],df_test_2540['pl_pred'])).mean()
RMSE_2540 = math.sqrt(MSE)

R2_2540= r2_score(df_test_2540['pl'],df_test_2540['pl_pred']) #R2
R2_oos_2540 = 1 - np.sum((df_test_2540['pl'] - df_test_2540['pl_pred'])**2) / np.sum((df_train_2540['pl'].mean() - df_test_2540['pl'])**2)

MSE = np.square(np.subtract(df_test_3500['pl'],df_test_3500['pl_pred'])).mean()
RMSE_3500 = math.sqrt(MSE)

R2_3500= r2_score(df_test_3500['pl'],df_test_3500['pl_pred']) #R2
R2_oos_3500 = 1 - np.sum((df_test_3500['pl'] - df_test_3500['pl_pred'])**2) / np.sum((df_train_3500['pl'].mean() - df_test_3500['pl'])**2)

#Testing

#variance
variance_735 = np.sum((df_test_735['pl'] - df_test_735['pl'].mean())**2)/len(df_test_735['pl'])
variance_2540 = np.sum((df_test_2540['pl'] - df_test_2540['pl'].mean())**2)/len(df_test_2540['pl'])
variance_3500 = np.sum((df_test_3500['pl'] - df_test_3500['pl'].mean())**2)/len(df_test_3500['pl'])

#mean
mean_735=np.mean(df_test_735['pl'])
mean_2540=np.mean(df_test_2540['pl'])
mean_3500=np.mean(df_test_3500['pl'])


data = [['750',RMSE_735, R2_735,R2_oos_735,variance_735,mean_735],['2500',RMSE_2540, R2_2540,R2_oos_2540,variance_2540,mean_2540],['3500',RMSE_3500, R2_3500,R2_oos_3500,variance_3500,mean_3500]]  
print(tabulate(data, headers=["Freq",'RMSE','R^2','R^2 OOS','Variance [dB]','Mean[dB]']))


#Plot
p1 = max(max(y_target_desn_test), max(y_target_desn_test))
p2 = min(min(y_target_desn_test), min(y_target_desn_test))
plt.plot([p1, p2], [p1, p2], '--',linewidth=1,label= 'Best fit',c='navy')

plt.scatter(y_target_desn_test,y_pred_desn_test,label='Result ($R^2:0.10$)')
plt.grid(ls='--')
plt.legend()
plt.xlabel('Measured path-loss [dB]')
plt.ylabel('Predicted path-loss [dB]')
plt.savefig('R2_macrocell.eps',format='eps',dpi=1200)
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

learning_rate = [0.001,0.01,0.1] #define learning rate

decay_weigth = [0.001,0.01,0.1]  #define decay weigth

max_iter = len(learning_rate)*len(decay_weigth)  #number of iteractions

iteration = list(range(1,max_iter+1))

num_neurons = np.arange(1, 21, 1) #number of neurons

p = len(num_neurons)*max_iter
zmatrix_mlp = np.zeros((p, 5)) #size of the cross-validation matriz

cv = KFold(n_splits=5,shuffle=True,random_state=0)


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
    
    for k in range(len(num_neurons)): 
        
        model = MLPRegressor(hidden_layer_sizes=num_neurons[k], 
                                 activation='relu',
                                 solver='lbfgs',
                                 alpha=decay_weigth[ml],
                                 learning_rate_init=learning_rate[md],
                                 momentum=0.9,
                                 random_state=0, 
                                 max_iter=6000,
                                 early_stopping=True)    
        

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
        zmatrix_mlp[count][0] = num_neurons[k]
        zmatrix_mlp[count][1] = learning_rate[md]
        zmatrix_mlp[count][2] = decay_weigth[ml]
        zmatrix_mlp[count][3] = RMSE_tot_train
        zmatrix_mlp[count][4] = RMSE_tot_val
        count += 1
        c +=1
        
      
        if c == a + len(num_neurons)+1:
            md +=1
            a = a + len(num_neurons)
            x +=1

        if x == len(learning_rate):
            x=0
            md=0
        
        if c == b + len(num_neurons)*len(learning_rate)+1:
            ml +=1
            b = b + len(num_neurons)*len(learning_rate)

