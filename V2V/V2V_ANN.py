# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import math
from sklearn.neural_network import MLPRegressor

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
Reading the CSV files
"""
path = r"Dataset_V2V.csv"
df = pd.read_csv(path)
df.head()

n=1000 #the first 1000 samples for the Route Jardim Oceanico
df_train= df.iloc[:n]

n=374#the last 374 samples for the Route Jardim Oceanico
df_test= df.tail(n)

#%%
"""
Create Input and Output Data
"""
X_train = df_train.iloc[:, [13,14,16,17]] #Predictors
y_train = df_train.iloc[:, [8]] #PL

X_test = df_test.iloc[:, [13,14,16,17]] 
y_test = df_test.iloc[:, [8]]

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

gtb_regressor =  MLPRegressor(hidden_layer_sizes=16, 
                                 activation='relu',
                                 solver='lbfgs',
                                 alpha=0.1,
                                 learning_rate_init=0.1,
                                 momentum=0.9,
                                 random_state=0, 
                                 max_iter=6000,
                                 early_stopping=True)    
gtb_regressor.fit(X_train,np.ravel(y_train))

#TRAINING
y_pred = gtb_regressor.predict(X_train)

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
y_pred_test = gtb_regressor.predict(X_test)
y_pred_test = y_pred_test.reshape(-1,1)

y_pred_desn_test = scaler.inverse_transform(y_pred_test)
y_target_desn_test = scaler.inverse_transform(y_test)

#Plot
p1 = max(max(y_target_desn_test), max(y_target_desn_test))
p2 = min(min(y_target_desn_test), min(y_target_desn_test))
plt.plot([p1, p2], [p1, p2], '--',linewidth=1,label= 'Best fit',c='navy')

plt.scatter(y_target_desn_test,y_pred_desn_test,label= 'GTB PL-based model')
plt.grid(ls='--')
plt.legend()
plt.xlabel('Measured path-loss [dB]')
plt.ylabel('Predicted path-loss [dB]')
plt.savefig('R2_V2I.eps',format='eps',dpi=1200)
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

"""
Applying Cross-Validation
"""

learning_rate = [0.001,0.01,0.1] #np.arange(2, 13, 1) #define maximum depth
    
decay_weigth = [0.001,0.01,0.1]  #define minumum samples leaf

max_iter = len(learning_rate)*len(decay_weigth)  #number of iteractions

iteration = list(range(1,max_iter+1))

num_neurons = np.arange(1, 21, 1) #number of trees

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
        
        #To change the maximum depth value
        if c == a + len(num_neurons)+1:
            md +=1
            a = a + len(num_neurons)
            x +=1
            
        #To change the minimum samples leaf value    
        if x == len(learning_rate):
            x=0
            md=0
        
        if c == b + len(num_neurons)*len(learning_rate)+1:
            ml +=1
            b = b + len(num_neurons)*len(learning_rate)
