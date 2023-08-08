# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
  
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import math
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")

import tabulate
from tabulate import tabulate

import random
random.seed(0)

"""
Reading the CSV files
"""
#Route1
path=r"Route1_735.csv"
df_R1_735 = pd.read_csv(path)
df_R1_735.head()

path=r"Route1_2540.csv"
df_R1_2540 = pd.read_csv(path)
df_R1_2540.head()

path=r"Route1_3500.csv"
df_R1_3500 = pd.read_csv(path)
df_R1_3500.head()

#Route2
path=r"Route2_735.csv"
df_R2_735 = pd.read_csv(path)
df_R2_735.head()

path=r"Route2_2540.csv"
df_R2_2540 = pd.read_csv(path)
df_R2_2540.head()

path=r"Route2_3500.csv"
df_R2_3500 = pd.read_csv(path)
df_R2_3500.head()

#%%
#Route1
df_train = pd.concat([
    df_R1_735,
    df_R1_2540, 
    df_R1_3500
    ])

#Route2
df_test = pd.concat([
    df_R2_735, 
    df_R2_2540, 
    df_R2_3500 
    ])

#%%
"""
Create Input and Output Data
"""
X_train = df_train.iloc[:, [6,15,11,17,12,13]] 
y_train = df_train.iloc[:, [25]]

X_test = df_test.iloc[:, [6,15,11,17,12,13]] 
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

rf_regressor = RandomForestRegressor(n_estimators=34, max_depth=2, min_samples_leaf=10,criterion='squared_error', random_state=42)
rf_regressor.fit(X_train,np.ravel(y_train))

#TRAINING
y_pred = rf_regressor.predict(X_train)

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
y_pred_test = rf_regressor.predict(X_test)
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
max_depth = np.arange(2, 5, 1) #np.arange(2, 13, 1) #define maximum depth
    
min_samples_leaf = np.arange(10, 24, 2)  #define minumum samples leaf

max_iter = len(max_depth)*len(min_samples_leaf)  #number of iteractions

iteration = list(range(1,max_iter+1))

num_trees = np.arange(2, 72, 2) #number of trees

p = len(num_trees)*max_iter
zmatrix_cv_rf = np.zeros((p, 5)) #size of the cross-validation matriz

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
    
    for k in range(len(num_trees)): 
        
        model = RandomForestRegressor(n_estimators=num_trees[k], max_depth=max_depth[md], min_samples_leaf=min_samples_leaf[ml],criterion='absolute_error', random_state=42)
    
        
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
        zmatrix_cv_rf[count][0] = num_trees[k]
        zmatrix_cv_rf[count][1] = max_depth[md]
        zmatrix_cv_rf[count][2] = min_samples_leaf[ml]
        zmatrix_cv_rf[count][3] = RMSE_tot_train
        zmatrix_cv_rf[count][4] = RMSE_tot_val
        count += 1
        c +=1
        
        #To change the maximum depth value
        if c == a + len(num_trees)+1:
            md +=1
            a = a + len(num_trees)
            x +=1
            
        #To change the minimum samples leaf value    
        if x == len(max_depth):
            x=0
            md=0
        
        if c == b + len(num_trees)*len(max_depth)+1:
            ml +=1
            b = b + len(num_trees)*len(max_depth)

