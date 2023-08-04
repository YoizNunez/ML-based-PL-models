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
from sklearn.ensemble import GradientBoostingRegressor

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

#SC1
path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\V2I\Final Dataset SC1\with std\SC1_750_Vegetation.csv"
df_SC1_750 = pd.read_csv(path)
df_SC1_750.head()

path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\V2I\Final Dataset SC1\with std\SC1_2500_Vegetation.csv"
df_SC1_2500 = pd.read_csv(path)
df_SC1_2500.head()


path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\V2I\Final Dataset SC1\with std\SC1_3500_Vegetation.csv"
df_SC1_3500 = pd.read_csv(path)
df_SC1_3500.head()

#SC2
path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\V2I\Final Dataset SC1\with std\SC2_750_Vegetation.csv"
df_SC2_750 = pd.read_csv(path)
df_SC2_750.head()

path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\V2I\Final Dataset SC1\with std\SC2_2500_Vegetation.csv"
df_SC2_2500 = pd.read_csv(path)
df_SC2_2500.head()


path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\V2I\Final Dataset SC1\with std\SC2_3500_Vegetation.csv"
df_SC2_3500 = pd.read_csv(path)
df_SC2_3500.head()

#%%

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

x_input=[12,5,7,10,4,15,9,8,18,20,19,16,17,13]

X_train = df_train.iloc[:, x_input] 
y_train = df_train.iloc[:, [21]]

#Select the testing set: SC1 or SC2
X_test = df_test.iloc[:, x_input] #12,5,7,10,4,15
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


gtb_regressor = GradientBoostingRegressor(n_estimators=70,
                                          learning_rate=0.1,
                                          max_depth=3, 
                                          min_samples_leaf=1, 
                                          alpha=0.9, 
                                          random_state=42, 
                                          loss='ls', 
                                          max_features='auto')
gtb_regressor.fit(X_train,np.ravel(y_train))


#TRAINING
y_pred = gtb_regressor.predict(X_train)

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
y_pred_test = gtb_regressor.predict(X_test)
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

plt.scatter(y_target_desn_test,y_pred_desn_test,label='Results for Route #1 ($R^2:0.44$)')
plt.grid(ls='--')
plt.legend()
plt.xlabel('Measured path loss [dB]')
plt.ylabel('Predicted path loss [dB]')
plt.savefig('R2_V2I_SC1.eps',format='eps',dpi=1200)
plt.show()
plt.close()


MSE = np.square(np.subtract(y_target_desn_test,y_pred_desn_test)).mean()
RMSE_test = math.sqrt(MSE)


R2_test= r2_score(y_target_desn_test,y_pred_desn_test) #R2

MAPE_test = np.mean(np.abs((y_target_desn_test - y_pred_desn_test)/y_target_desn_test))*100 #MAPE

n = len(y_pred_desn_test)
sum_model=0

abs_dif = np.abs(y_target_desn_test - y_pred_desn_test)
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

#IML Techniques
import pymint #version: 0.2.6
import pickle

#name of predictors
features_list=['bd',
               #'d',
               #'f',
               #'gh',
               #'h',
               #'vd',
               #'avg_vh',
               #'avg_bh',
               #'sigma_b',
               #'sigma_g',
               #'sigma_v',
               #'n_v',
               #'dif_bh',
               #'n_b'
               ]
   
X_train_df = pd.DataFrame(X_train, columns = features_list)
   
gtb_regressor = GradientBoostingRegressor(n_estimators=70, learning_rate=0.1, max_depth=3, min_samples_leaf=3, alpha=0.9, random_state=42, loss='ls', max_features='auto')
gtb_regressor.fit(X_train, np.ravel(y_train))
        
# saving the model as pickle file to be reading by the library pymint, according its especification.
pickle.dump(gtb_regressor, open('gtbmodel_IML.pkl','wb')) #save the model, i.e., 'mlpmodel_IML_5.pkl' for the subset of 5 predictors
gtb = pickle.load(open('gtbmodel_IML.pkl','rb')) #the file is load from the local path. In my case from C:/Users/Yoiz Nuñez
gtb_model = ('GTB',gtb)

#ALE
explainer_gtb = pymint.InterpretToolkit(gtb_model,X=X_train_df, y=y_train)
ale_model = explainer_gtb.ale(features=features_list, n_bins=30, n_jobs=1, subsample=1.0, n_bootstrap=1)

MEC = explainer_gtb.main_effect_complexity(ale_model,max_segments=10)

IAS= explainer_gtb.interaction_strength(ale_model)
#%%

"All predictors"
#predictor dist

ale_bd_value_s6 = ale_model.bd__GTB__ale.to_masked_array()  
ale_bd_value_s6= ale_bd_value_s6.ravel() #to reduce 1 dim
ale_bd_bin_s6= ale_model.bd__bin_values.to_masked_array()

ale_d_value_s6 = ale_model.d__GTB__ale.to_masked_array()  
ale_d_value_s6= ale_d_value_s6.ravel() #to reduce 1 dim
ale_d_bin_s6= ale_model.d__bin_values.to_masked_array()

ale_f_value_s6 = ale_model.f__GTB__ale.to_masked_array()  
ale_f_value_s6= ale_f_value_s6.ravel() #to reduce 1 dim
ale_f_bin_s6= ale_model.f__bin_values.to_masked_array()

ale_gh_value_s6 = ale_model.gh__GTB__ale.to_masked_array()  
ale_gh_value_s6= ale_gh_value_s6.ravel() #to reduce 1 dim
ale_gh_bin_s6= ale_model.gh__bin_values.to_masked_array()

ale_h_value_s6 = ale_model.h__GTB__ale.to_masked_array()  
ale_h_value_s6= ale_h_value_s6.ravel() #to reduce 1 dim
ale_h_bin_s6= ale_model.h__bin_values.to_masked_array()

ale_vd_value_s6 = ale_model.vd__GTB__ale.to_masked_array()  
ale_vd_value_s6= ale_vd_value_s6.ravel() #to reduce 1 dim
ale_vd_bin_s6= ale_model.vd__bin_values.to_masked_array()

#%%

"5 pred"
ale_bd_value_s5 = ale_model.bd__GTB__ale.to_masked_array()  
ale_bd_value_s5= ale_bd_value_s5.ravel() #to reduce 1 dim
ale_bd_bin_s5= ale_model.bd__bin_values.to_masked_array()

ale_d_value_s5 = ale_model.d__GTB__ale.to_masked_array()  
ale_d_value_s5= ale_d_value_s5.ravel() #to reduce 1 dim
ale_d_bin_s5= ale_model.d__bin_values.to_masked_array()

ale_f_value_s5 = ale_model.f__GTB__ale.to_masked_array()  
ale_f_value_s5= ale_f_value_s5.ravel() #to reduce 1 dim
ale_f_bin_s5= ale_model.f__bin_values.to_masked_array()

ale_gh_value_s5 = ale_model.gh__GTB__ale.to_masked_array()  
ale_gh_value_s5= ale_gh_value_s5.ravel() #to reduce 1 dim
ale_gh_bin_s5= ale_model.gh__bin_values.to_masked_array()

ale_h_value_s5 = ale_model.h__GTB__ale.to_masked_array()  
ale_h_value_s5= ale_h_value_s5.ravel() #to reduce 1 dim
ale_h_bin_s5= ale_model.h__bin_values.to_masked_array()

#%%

"4-pred"
ale_bd_value_s4 = ale_model.bd__GTB__ale.to_masked_array()  
ale_bd_value_s4= ale_bd_value_s4.ravel() #to reduce 1 dim
ale_bd_bin_s4= ale_model.bd__bin_values.to_masked_array()

ale_d_value_s4 = ale_model.d__GTB__ale.to_masked_array()  
ale_d_value_s4= ale_d_value_s4.ravel() #to reduce 1 dim
ale_d_bin_s4= ale_model.d__bin_values.to_masked_array()

ale_f_value_s4 = ale_model.f__GTB__ale.to_masked_array()  
ale_f_value_s4= ale_f_value_s4.ravel() #to reduce 1 dim
ale_f_bin_s4= ale_model.f__bin_values.to_masked_array()

ale_gh_value_s4 = ale_model.gh__GTB__ale.to_masked_array()  
ale_gh_value_s4= ale_gh_value_s4.ravel() #to reduce 1 dim
ale_gh_bin_s4= ale_model.gh__bin_values.to_masked_array()

#%%

"3-pred"
ale_bd_value_s3 = ale_model.bd__GTB__ale.to_masked_array()  
ale_bd_value_s3= ale_bd_value_s3.ravel() #to reduce 1 dim
ale_bd_bin_s3= ale_model.bd__bin_values.to_masked_array()

ale_d_value_s3 = ale_model.d__GTB__ale.to_masked_array()  
ale_d_value_s3= ale_d_value_s3.ravel() #to reduce 1 dim
ale_d_bin_s3= ale_model.d__bin_values.to_masked_array()

ale_f_value_s3 = ale_model.f__GTB__ale.to_masked_array()  
ale_f_value_s3= ale_f_value_s3.ravel() #to reduce 1 dim
ale_f_bin_s3= ale_model.f__bin_values.to_masked_array()

#%%

"2-pred"

ale_bd_value_s2 = ale_model.bd__GTB__ale.to_masked_array()  
ale_bd_value_s2= ale_bd_value_s2.ravel() #to reduce 1 dim
ale_bd_bin_s2= ale_model.bd__bin_values.to_masked_array()

ale_d_value_s2 = ale_model.d__GTB__ale.to_masked_array()  
ale_d_value_s2= ale_d_value_s2.ravel() #to reduce 1 dim
ale_d_bin_s2= ale_model.d__bin_values.to_masked_array()


#%%

"1-pred"
ale_bd_value_s1 = ale_model.bd__GTB__ale.to_masked_array()  
ale_bd_value_s1= ale_bd_value_s1.ravel() #to reduce 1 dim
ale_bd_bin_s1= ale_model.bd__bin_values.to_masked_array()


#%%

fig = plt.figure(figsize=(10,6))
spec = gridspec.GridSpec(ncols=3, nrows=2) # 6 columns evenly divides both 2 & 3

ax1 = fig.add_subplot(spec[0,0]) # row 0 with axes spanning 2 cols on evens
ax2 = fig.add_subplot(spec[0,1]) # row 0 with axes spanning 2 cols on evens
ax3 = fig.add_subplot(spec[0,2]) # row 0 with axes spanning 2 cols on evens
ax4 = fig.add_subplot(spec[1,0]) # row 0 with axes spanning 2 cols on evens
ax5 = fig.add_subplot(spec[1,1]) # row 0 with axes spanning 2 cols on evens
ax6 = fig.add_subplot(spec[1,2]) # row 0 with axes spanning 2 cols on evens

#bd
ax1.plot(ale_bd_bin_s1,ale_bd_value_s1,"+-", label ='1',lw=3,color='blue')
ax1.plot(ale_bd_bin_s2,ale_bd_value_s2,"-.", label ='2',lw=3,color='red')
ax1.plot(ale_bd_bin_s3,ale_bd_value_s3,"--", label ='3',lw=3,color='green')
ax1.plot(ale_bd_bin_s4,ale_bd_value_s4,":",label ='4',lw=5,color='orange')
ax1.plot(ale_bd_bin_s5,ale_bd_value_s5,"x-", label ='5',lw=3,color='brown')
ax1.plot(ale_bd_bin_s6,ale_bd_value_s6, label ='6',lw=3,color='purple')
ax1.grid(ls='--')
ax1.set_xlabel('$b_{depth}$ (m)',fontsize=18)
ax1.set_ylabel('(a)',fontsize=18)
ax1.xaxis.set_tick_params(labelsize=16)
ax1.yaxis.set_tick_params(labelsize=16)
ax1.legend(fancybox=False,shadow=False,fontsize="13")
x_labels1 = ['','54','108','162','216','271']
ax1.set_xticklabels(x_labels1)
    

#d
ax2.plot(ale_d_bin_s2,ale_d_value_s2,"-.", label ='2',lw=3,color='red')
ax2.plot(ale_d_bin_s3,ale_d_value_s3,"--", label ='3',lw=3,color='green')
ax2.plot(ale_d_bin_s4,ale_d_value_s4,":", label ='4',lw=5,color='orange')
ax2.plot(ale_d_bin_s5,ale_d_value_s5,"x-", label ='5',lw=3,color='brown')
ax2.plot(ale_d_bin_s6,ale_d_value_s6, label ='6',lw=3,color='purple')
ax2.grid(ls='--')
ax2.set_xlabel('$d$ (m)',fontsize=18)
ax2.set_ylabel('(b)',fontsize=18)
ax2.xaxis.set_tick_params(labelsize=16)
ax2.yaxis.set_tick_params(labelsize=16)
ax2.legend(fancybox=False,shadow=False,fontsize="13")
x_labels2 = ['27','30','33','36','40']
ax2.set_xticklabels(x_labels2)


#f
ax3.plot(ale_f_bin_s3,ale_f_value_s3,"--", label ='3',lw=3,color='green')
ax3.plot(ale_f_bin_s4,ale_f_value_s4,":", label ='4',lw=5,color='orange')
ax3.plot(ale_f_bin_s5,ale_f_value_s5,"x-", label ='5',lw=4,color='brown')
ax3.plot(ale_f_bin_s6,ale_f_value_s6, label ='6',lw=4,color='purple')
ax3.grid(ls='--')
ax3.set_xlabel('f (GHz)',fontsize=18)
ax3.set_ylabel('(c)',fontsize=18)
ax3.xaxis.set_tick_params(labelsize=16)
ax3.yaxis.set_tick_params(labelsize=16)
ax3.legend(fancybox=False,shadow=False,fontsize="13")
x_labels3 = ['0','1.4','2.7','4.06','5.4']
ax3.set_xticklabels(x_labels3)

#gh
ax4.plot(ale_gh_bin_s4,ale_gh_value_s4,":", label ='4',lw=5,color='orange')
ax4.plot(ale_gh_bin_s5,ale_gh_value_s5,"x-", label ='5',lw=3,color='brown')
ax4.plot(ale_gh_bin_s6,ale_gh_value_s6, label ='6',lw=3,color='purple')
ax4.grid(ls='--')
ax4.set_xlabel('$\overline{g}_h$ (m)',fontsize=18)
ax4.set_ylabel('(d)',fontsize=18)
ax4.xaxis.set_tick_params(labelsize=16)
ax4.yaxis.set_tick_params(labelsize=16)
ax4.legend(fancybox=False,shadow=False,fontsize="13")
x_labels4 = ['15','24.5','34.0','43.5','53.0']
ax4.set_xticklabels(x_labels4)

#gh
ax5.plot(ale_h_bin_s5,ale_h_value_s5,"x-", label ='5',lw=3,color='brown')
ax5.plot(ale_h_bin_s6,ale_h_value_s6, label ='6',lw=3,color='purple')
ax5.grid(ls='--')
ax5.set_xlabel('$\Delta_h$ (m)',fontsize=18)
ax5.set_ylabel('(e)',fontsize=18)
ax5.xaxis.set_tick_params(labelsize=16)
ax5.yaxis.set_tick_params(labelsize=16)
ax5.legend(fancybox=False,shadow=False,fontsize="13")
x_labels4 = ['15','24.5','34.0','43.5','53.0']
ax4.set_xticklabels(x_labels4)

#vd
ax6.plot(ale_vd_bin_s6,ale_vd_value_s6, label ='6',lw=3,color='purple')
ax6.grid(ls='--')
ax6.set_xlabel('$v_{depth}(m)$',fontsize=18)
ax6.set_ylabel('(f)',fontsize=18)
ax6.xaxis.set_tick_params(labelsize=16)
ax6.yaxis.set_tick_params(labelsize=16)
ax6.legend(fancybox=False,shadow=False,fontsize="13")
x_labels6 = ['15','24.5','34.0','43.5','53.0']
ax6.set_xticklabels(x_labels6)

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.5
                    )

plt.savefig('ALE-GTB-macro-cov.png',format='png',dpi=600)
plt.show()
plt.close()

