# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import math
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")

import numpy
import matplotlib.pyplot as plt

import tabulate
from tabulate import tabulate

import random
random.seed(0)

#%%
"""
Read Data
"""
path=r"outdoor.csv"
df = pd.read_csv(path)
df.head()

#Testing Rx
df_train= df.loc[
                (df['RX']=='RX1')|  
                (df['RX']=='RX2') | 
                (df['RX']=='RX4') | 
                (df['RX']=='RX6') | 
                (df['RX']=='RX7') | 
                (df['RX']=='RX8') | 
                #(df['RX']=='RX10') | 
                (df['RX']=='RX9') | 
                (df['RX']=='RX10') | 
                #(df['RX']=='RX14')  | 
                (df['RX']=='RX11')  | 
                (df['RX']=='RX14')  | 
                (df['RX']=='RX16')  | 
                (df['RX']=='RX17')  | 
                (df['RX']=='RX18')  | 
                (df['RX']=='RX19')  | 
                (df['RX']=='RX21')  | 
                (df['RX']=='RX22')  | 
                (df['RX']=='RX23')   
                #(df['RX']=='RX21') | 
                #(df['RX']=='RX22')  
                #(df['RX']=='RX23')  
                             
                ] 

df_test= df.loc[(df['RX']=='RX5')  | 
                (df['RX']=='RX12')  | 
                (df['RX']=='RX15') |
                (df['RX']=='RX20') 
                ]

#%%
X_train = df_train.iloc[:, [1,0,5,2]] 
y_train = df_train.iloc[:, [3]]

X_test = df_test.iloc[:, [1,0,5,2]] 
y_test = df_test.iloc[:, [3]]

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

gtb_regressor = GradientBoostingRegressor(n_estimators=60, 
                                          learning_rate=0.1,
                                          max_depth=1, 
                                          min_samples_leaf=3, 
                                          alpha=0.9, 
                                          random_state=80, 
                                          loss='absolute_error', 
                                          max_features='auto')
gtb_regressor.fit(X_train,np.ravel(y_train))

#%%
#TRAINING
y_pred = gtb_regressor.predict(X_train)

y_pred = y_pred.reshape(-1,1)

y_pred_desn = scaler.inverse_transform(y_pred)
y_target_desn = scaler.inverse_transform(y_train)

#%%
MSE = np.square(np.subtract(y_target_desn,y_pred_desn)).mean() #RMSE
RMSE_train = math.sqrt(MSE)

MAPE_train = np.mean(np.abs((y_target_desn - y_pred_desn)/y_target_desn))*100 #MAPE

R2_train= r2_score(y_target_desn,y_pred_desn) #R2

n = len(y_pred_desn)
sum_model=0

abs_dif = np.abs(y_target_desn,y_pred_desn)
mean_model = np.mean(abs_dif)

for x in abs_dif:
    t = (x - mean_model) ** 2
    sum_model += t 

SD_train = math.sqrt(sum_model/(n)) #SD

#%%
#TESTING
y_pred_test = gtb_regressor.predict(X_test)
y_pred_test = y_pred_test.reshape(-1,1)

y_pred_desn_test = scaler.inverse_transform(y_pred_test)
y_target_desn_test = scaler.inverse_transform(y_test)

#%%
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

abs_dif = np.abs(y_target_desn_test,y_pred_desn_test)
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

max_depth = list(range(3,7)) #define maximum depth
    
min_samples_leaf = list(range(1,4)) #define minumum samples leaf

max_iter = len(max_depth)*len(min_samples_leaf)  #number of iteractions

iteration = list(range(1,max_iter+1))

num_trees = np.arange(8, 256, 17) #number of trees

p = len(num_trees)*max_iter
matrix_cv_gb = np.zeros((p, 5)) #size of the cross-validation matriz

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

#iterations
for i in range(max_iter):
    
    for k in range(len(num_trees)): 
        
        model = GradientBoostingRegressor(n_estimators=num_trees[k], learning_rate=0.1, max_depth=max_depth[md], min_samples_leaf=min_samples_leaf[ml], alpha=0.9, random_state=42, loss='ls', max_features='auto')
        
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
        matrix_cv_gb[count][0] = num_trees[k]
        matrix_cv_gb[count][1] = max_depth[md]
        matrix_cv_gb[count][2] = min_samples_leaf[ml]
        matrix_cv_gb[count][3] = RMSE_tot_train
        matrix_cv_gb[count][4] = RMSE_tot_val
        count += 1
        c +=1
        
        if c == a + len(num_trees)+1:
            md +=1
            a = a + len(num_trees)
            x +=1
            
        if x == 4:
            x=0
            md=0
        
        if c == b + len(num_trees)*len(max_depth)+1:
            ml +=1
            b = b + len(num_trees)*len(max_depth)

for x in abs_dif:
    t = (x - mean_model) ** 2
    sum_model += t 

SD = math.sqrt(sum_model/(n)) #SD
print(SD)
