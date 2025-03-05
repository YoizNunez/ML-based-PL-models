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

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")

import numpy
import matplotlib.pyplot as plt

import tabulate
from tabulate import tabulate

# check xgboost version
import xgboost
from xgboost import XGBRegressor

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

x=[6,15,11,17,12,13,7#,14,10,8,9,16
  ] #Predictors

X_train = df_train.iloc[:, x]
y_train = df_train.iloc[:, [25]] #PL

X_test = df_test.iloc[:, x]
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


gtb_regressor = GradientBoostingRegressor(n_estimators=70, 
                                          learning_rate=0.14, 
                                          max_depth=10, 
                                          min_samples_leaf=14,
                                          alpha=0.9,
                                          loss='absolute_error', 
                                          max_features='auto',random_state=42) #42
gtb_regressor.fit(X_train,np.ravel(y_train))


#TRAINING
y_pred = gtb_regressor.predict(X_train)

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
data = [['735',RMSE_735, R2_735,variance_735,mean_735]
        ,['2450',RMSE_2540, R2_2540,variance_2540,mean_2540]
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
y_pred_test = gtb_regressor.predict(X_test)
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

data = [['735',RMSE_735, R2_735,R2_oos_735,variance_735,mean_735]
        ,['2450',RMSE_2540, R2_2540,R2_oos_2540,variance_2540,mean_2540]
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
import pymint #version: 0.2.6
import pickle

#name of predictors
features_list=['f',
               'nb',
               'sb',
               'nv',
               'sv',
               'sg',
               'd',
               'bd',
               'ag',
               'ab',
               'av',
               'vd'
               ]
   
X_train_df = pd.DataFrame(X_train, columns = features_list)
   
gtb_regressor = GradientBoostingRegressor(n_estimators=70, learning_rate=0.14, max_depth=8, min_samples_leaf=10, alpha=0.9, random_state=42, loss='absolute_error', max_features='auto')
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

ale_f_value_s7 = ale_model.f__GTB__ale.to_masked_array()  
ale_f_value_s7= ale_f_value_s7.ravel() #to reduce 1 dim
ale_f_bin_s7= ale_model.f__bin_values.to_masked_array()

ale_nb_value_s7 = ale_model.nb__GTB__ale.to_masked_array()  
ale_nb_value_s7= ale_nb_value_s7.ravel() #to reduce 1 dim
ale_nb_bin_s7= ale_model.nb__bin_values.to_masked_array()

ale_sb_value_s7 = ale_model.sb__GTB__ale.to_masked_array()  
ale_sb_value_s7= ale_sb_value_s7.ravel() #to reduce 1 dim
ale_sb_bin_s7= ale_model.sb__bin_values.to_masked_array()

ale_nv_value_s7 = ale_model.nv__GTB__ale.to_masked_array()  
ale_nv_value_s7= ale_nv_value_s7.ravel() #to reduce 1 dim
ale_nv_bin_s7= ale_model.nv__bin_values.to_masked_array()

ale_sv_value_s7 = ale_model.sv__GTB__ale.to_masked_array()  
ale_sv_value_s7= ale_sv_value_s7.ravel() #to reduce 1 dim
ale_sv_bin_s7= ale_model.sv__bin_values.to_masked_array()

ale_sg_value_s7 = ale_model.sg__GTB__ale.to_masked_array()  
ale_sg_value_s7= ale_sg_value_s7.ravel() #to reduce 1 dim
ale_sg_bin_s7= ale_model.sg__bin_values.to_masked_array()

ale_d_value_s7 = ale_model.d__GTB__ale.to_masked_array()  
ale_d_value_s7= ale_d_value_s7.ravel() #to reduce 1 dim
ale_d_bin_s7= ale_model.d__bin_values.to_masked_array()

#%%
"6-pred"

ale_f_value_s6 = ale_model.f__GTB__ale.to_masked_array()  
ale_f_value_s6= ale_f_value_s6.ravel() #to reduce 1 dim
ale_f_bin_s6= ale_model.f__bin_values.to_masked_array()

ale_nb_value_s6 = ale_model.nb__GTB__ale.to_masked_array()  
ale_nb_value_s6= ale_nb_value_s6.ravel() #to reduce 1 dim
ale_nb_bin_s6= ale_model.nb__bin_values.to_masked_array()

ale_sb_value_s6 = ale_model.sb__GTB__ale.to_masked_array()  
ale_sb_value_s6= ale_sb_value_s6.ravel() #to reduce 1 dim
ale_sb_bin_s6= ale_model.sb__bin_values.to_masked_array()

ale_nv_value_s6 = ale_model.nv__GTB__ale.to_masked_array()  
ale_nv_value_s6= ale_nv_value_s6.ravel() #to reduce 1 dim
ale_nv_bin_s6= ale_model.nv__bin_values.to_masked_array()

ale_sv_value_s6 = ale_model.sv__GTB__ale.to_masked_array()  
ale_sv_value_s6= ale_sv_value_s6.ravel() #to reduce 1 dim
ale_sv_bin_s6= ale_model.sv__bin_values.to_masked_array()

ale_sg_value_s6 = ale_model.sg__GTB__ale.to_masked_array()  
ale_sg_value_s6= ale_sg_value_s6.ravel() #to reduce 1 dim
ale_sg_bin_s6= ale_model.sg__bin_values.to_masked_array()

#%%

"5-pred"
ale_f_value_s5 = ale_model.f__GTB__ale.to_masked_array()  
ale_f_value_s5= ale_f_value_s5.ravel() #to reduce 1 dim
ale_f_bin_s5= ale_model.f__bin_values.to_masked_array()

ale_nb_value_s5 = ale_model.nb__GTB__ale.to_masked_array()  
ale_nb_value_s5= ale_nb_value_s5.ravel() #to reduce 1 dim
ale_nb_bin_s5= ale_model.nb__bin_values.to_masked_array()

ale_sb_value_s5 = ale_model.sb__GTB__ale.to_masked_array()  
ale_sb_value_s5= ale_sb_value_s5.ravel() #to reduce 1 dim
ale_sb_bin_s5= ale_model.sb__bin_values.to_masked_array()

ale_nv_value_s5 = ale_model.nv__GTB__ale.to_masked_array()  
ale_nv_value_s5= ale_nv_value_s5.ravel() #to reduce 1 dim
ale_nv_bin_s5= ale_model.nv__bin_values.to_masked_array()

ale_sv_value_s5 = ale_model.sv__GTB__ale.to_masked_array()  
ale_sv_value_s5= ale_sv_value_s5.ravel() #to reduce 1 dim
ale_sv_bin_s5= ale_model.sv__bin_values.to_masked_array()


#%%

"4-pred"
ale_f_value_s4 = ale_model.f__GTB__ale.to_masked_array()  
ale_f_value_s4= ale_f_value_s4.ravel() #to reduce 1 dim
ale_f_bin_s4= ale_model.f__bin_values.to_masked_array()

ale_nb_value_s4 = ale_model.nb__GTB__ale.to_masked_array()  
ale_nb_value_s4= ale_nb_value_s4.ravel() #to reduce 1 dim
ale_nb_bin_s4= ale_model.nb__bin_values.to_masked_array()

ale_sb_value_s4 = ale_model.sb__GTB__ale.to_masked_array()  
ale_sb_value_s4= ale_sb_value_s4.ravel() #to reduce 1 dim
ale_sb_bin_s4= ale_model.sb__bin_values.to_masked_array()

ale_nv_value_s4 = ale_model.nv__GTB__ale.to_masked_array()  
ale_nv_value_s4= ale_nv_value_s4.ravel() #to reduce 1 dim
ale_nv_bin_s4= ale_model.nv__bin_values.to_masked_array()


#%%

"3-pred"
ale_f_value_s3 = ale_model.f__GTB__ale.to_masked_array()  
ale_f_value_s3= ale_f_value_s3.ravel() #to reduce 1 dim
ale_f_bin_s3= ale_model.f__bin_values.to_masked_array()

ale_nb_value_s3 = ale_model.nb__GTB__ale.to_masked_array()  
ale_nb_value_s3= ale_nb_value_s3.ravel() #to reduce 1 dim
ale_nb_bin_s3= ale_model.nb__bin_values.to_masked_array()

ale_sb_value_s3 = ale_model.sb__GTB__ale.to_masked_array()  
ale_sb_value_s3= ale_sb_value_s3.ravel() #to reduce 1 dim
ale_sb_bin_s3= ale_model.sb__bin_values.to_masked_array()

#%%

"2-pred"
ale_f_value_s2 = ale_model.f__GTB__ale.to_masked_array()  
ale_f_value_s2= ale_f_value_s2.ravel() #to reduce 1 dim
ale_f_bin_s2= ale_model.f__bin_values.to_masked_array()

ale_nb_value_s2 = ale_model.nb__GTB__ale.to_masked_array()  
ale_nb_value_s2= ale_nb_value_s2.ravel() #to reduce 1 dim
ale_nb_bin_s2= ale_model.nb__bin_values.to_masked_array()

#%%

"1-pred"
ale_f_value_s1 = ale_model.f__GTB__ale.to_masked_array()  
ale_f_value_s1= ale_f_value_s1.ravel() #to reduce 1 dim
ale_f_bin_s1= ale_model.f__bin_values.to_masked_array()

#%%

import matplotlib.gridspec as gridspec


fig = plt.figure(figsize=(25,20))
spec = gridspec.GridSpec(ncols=3, nrows=3) # 6 columns evenly divides both 2 & 3

ax1 = fig.add_subplot(spec[0,0]) # row 0 with axes spanning 2 cols on evens
ax2 = fig.add_subplot(spec[0,1]) # row 0 with axes spanning 2 cols on evens
ax3 = fig.add_subplot(spec[0,2]) # row 0 with axes spanning 2 cols on evens
ax4 = fig.add_subplot(spec[1,0]) # row 0 with axes spanning 2 cols on evens
ax5 = fig.add_subplot(spec[1,1]) # row 0 with axes spanning 2 cols on evens
ax6 = fig.add_subplot(spec[1,2]) # row 0 with axes spanning 2 cols on evens
ax7 = fig.add_subplot(spec[2,0]) # row 0 with axes spanning 2 cols on evens

path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\Plot - ALE - V2I\ALE-f.csv"
df = pd.read_csv(path)

#f
ax1.plot(df['ale_f_bin_s1'],df['ale_f_value_s1'],"+-", label ='1',lw=6,color='blue')
ax1.plot(df['ale_f_bin_s2'],df['ale_f_value_s2'],"-.", label ='2',lw=6,color='red')
ax1.plot(df['ale_f_bin_s3'],df['ale_f_value_s3'],"--", label ='3',lw=6,color='green')
ax1.plot(df['ale_f_bin_s4'],df['ale_f_value_s4'],":",label ='4',lw=6,color='orange')
ax1.plot(df['ale_f_bin_s5'],df['ale_f_value_s5'],"x-", label ='5',lw=6,color='brown')
ax1.plot(df['ale_f_bin_s6'],df['ale_f_value_s6'], label ='6',lw=6,color='purple')
ax1.plot(df['ale_f_bin_s7'],df['ale_f_value_s7'], label ='7',lw=6,color='black')
ax1.grid(ls='--')
ax1.set_xlabel('$f$ (GHz)',fontsize=32)
ax1.set_title('(a)',size=32)
#ax1.set_ylabel('(a)',fontsize=18)
ax1.xaxis.set_tick_params(labelsize=32)
ax1.yaxis.set_tick_params(labelsize=32)
ax1.legend(fancybox=False,shadow=False,fontsize="20")
x_labels1 = ['','0.73','2.5','3.5']
ax1.set_xticklabels(x_labels1)
    
path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\Plot - ALE - V2I\ALE-nb.csv"
df = pd.read_csv(path)

#nb
ax2.plot(df['ale_nb_bin_s2'],df['ale_nb_value_s2'],"-.", label ='2',lw=6,color='red')
ax2.plot(df['ale_nb_bin_s3'],df['ale_nb_value_s3'],"-.", label ='3',lw=6,color='green')
ax2.plot(df['ale_nb_bin_s4'],df['ale_nb_value_s4'],"--", label ='4',lw=6,color='orange')
ax2.plot(df['ale_nb_bin_s5'],df['ale_nb_value_s5'],":", label ='5',lw=6,color='brown')
ax2.plot(df['ale_nb_bin_s6'],df['ale_nb_value_s6'],"x-", label ='6',lw=6,color='purple')
ax2.plot(df['ale_nb_bin_s7'],df['ale_nb_value_s7'], label ='7',lw=6,color='black')
ax2.set_title('(b)',size=32)
ax2.grid(ls='--')
ax2.set_xlabel('$n_b$',fontsize=32)
#ax2.set_ylabel('(b)',fontsize=18)
ax2.xaxis.set_tick_params(labelsize=32)
ax2.yaxis.set_tick_params(labelsize=32)
ax2.legend(fancybox=False,shadow=False,fontsize="20")
x_labels2 = ['4','8','12','16']
ax2.set_xticklabels(x_labels2)

path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\Plot - ALE - V2I\ALE-sb.csv"
df = pd.read_csv(path)
#sb
ax3.plot(df['ale_sb_bin_s3'],df['ale_sb_value_s3'],"--", label ='3',lw=6,color='green')
ax3.plot(df['ale_sb_bin_s4'],df['ale_sb_value_s4'],"--", label ='4',lw=6,color='orange')
ax3.plot(df['ale_sb_bin_s5'],df['ale_sb_value_s5'],":", label ='5',lw=6,color='brown')
ax3.plot(df['ale_sb_bin_s6'],df['ale_sb_value_s6'],"x-", label ='6',lw=6,color='purple')
ax3.plot(df['ale_sb_bin_s7'],df['ale_sb_value_s7'], label ='7',lw=6,color='black')
ax3.grid(ls='--')
ax3.set_xlabel('$\sigma_b$ (m)',fontsize=32)
ax3.set_title('(c)',size=32)
#ax3.set_ylabel('(c)',fontsize=18)
ax3.xaxis.set_tick_params(labelsize=32)
ax3.yaxis.set_tick_params(labelsize=32)
ax3.legend(fancybox=False,shadow=False,fontsize="20")
x_labels3 = ['2.17','6.29','10.41','14.54','18.66']
ax3.set_xticklabels(x_labels3)

path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\Plot - ALE - V2I\ALE-nv.csv"
df = pd.read_csv(path)

ax4.plot(df['ale_nv_bin_s4'],df['ale_nv_value_s4'],":", label ='4',lw=6,color='orange')
ax4.plot(df['ale_nv_bin_s5'],df['ale_nv_value_s5'],":", label ='5',lw=6,color='brown')
ax4.plot(df['ale_nv_bin_s6'],df['ale_nv_value_s6'],"x-", label ='6',lw=6,color='purple')
ax4.plot(df['ale_nv_bin_s7'],df['ale_nv_value_s7'], label ='7',lw=6,color='black')
ax4.grid(ls='--')
ax4.set_title('(d)',size=32)
ax4.set_xlabel('$n_v$',fontsize=32)
#ax4.set_ylabel('(d)',fontsize=18)
ax4.xaxis.set_tick_params(labelsize=32)
ax4.yaxis.set_tick_params(labelsize=32)
ax4.legend(fancybox=False,shadow=False,fontsize="20")
x_labels4 = ['0','2','4','6','8']
ax4.set_xticklabels(x_labels4)

path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\Plot - ALE - V2I\ALE-sv.csv"
df = pd.read_csv(path)

ax5.plot(df['ale_sv_bin_s5'],df['ale_sv_value_s5'],"x-", label ='5',lw=6,color='brown')
ax5.plot(df['ale_sv_bin_s6'],df['ale_sv_value_s6'],"x-", label ='6',lw=6,color='purple')
ax5.plot(df['ale_sv_bin_s7'],df['ale_sv_value_s7'], label ='7',lw=6,color='black')
ax5.grid(ls='--')
ax5.set_xlabel('$\sigma_v$ (m)',fontsize=32)
#ax5.set_ylabel('(e)',fontsize=18)
ax5.set_title('(e)',size=32)
ax5.xaxis.set_tick_params(labelsize=32)
ax5.yaxis.set_tick_params(labelsize=32)
ax5.legend(fancybox=False,shadow=False,fontsize="20")
x_labels5 = ['0','2.95','5.90','8.85','11.81']
ax5.set_xticklabels(x_labels5)


path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\Plot - ALE - V2I\ALE-sg.csv"
df = pd.read_csv(path)

ax6.plot(df['ale_sg_bin_s6'],df['ale_sg_value_s6'], label ='6',lw=6,color='purple')
ax6.plot(df['ale_sg_bin_s7'],df['ale_sg_value_s7'], label ='7',lw=6,color='black')
ax6.grid(ls='--')
ax6.set_xlabel('$\sigma_g$ (m)',fontsize=32)
#ax6.set_ylabel('(f)',fontsize=18)
ax6.xaxis.set_tick_params(labelsize=32)
ax6.yaxis.set_tick_params(labelsize=32)
ax6.set_title('(f)',size=32)
ax6.legend(fancybox=False,shadow=False,fontsize="20")
x_labels6 = ['0.82','1.63','2.45','3.26']
ax6.set_xticklabels(x_labels6)


path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\Plot - ALE - V2I\ALE-d.csv"
df = pd.read_csv(path)

ax7.plot(df['ale_d_bin_s7'],df['ale_d_value_s7'], label ='7',lw=6,color='black')
ax7.grid(ls='--')
ax7.set_xlabel('$d$ (m)',fontsize=32)
ax7.xaxis.set_tick_params(labelsize=32)
ax7.yaxis.set_tick_params(labelsize=32)
ax7.legend(fancybox=False,shadow=False,fontsize="20")
ax7.set_title('(g)',size=32)
#x_labels4 = ['15','24.5','34.0','43.5','53.0']
#ax4.set_xticklabels(x_labels4)
x_labels7 = ['69.8','138.4','206.9','275.5']
ax7.set_xticklabels(x_labels7)


plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.45
                    )

#fig.tight_layout()
plt.savefig('ALE-GTB-V2I.eps',format='eps',dpi=600)
plt.show()
plt.close()


#%%

#PFI
from sklearn.inspection import permutation_importance

r = permutation_importance(gtb_regressor, X_train, y_train,
                           n_repeats=10,
                           random_state=0)


X_train = df_train.iloc[:, [6,15,11,17,12,13,7]] 
y_train = df_train.iloc[:, [25]]

#Select the testing set: SC1 or SC2
X_test = df_test.iloc[:, [6,15,11,17,12,13,7]] 
y_test = df_test.iloc[:, [25]]

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

y_pred_train=gtb_regressor.predict(X_train)

# desnormalization
y_pred = y_pred_train.reshape(-1,1)
y_pred = scaler.inverse_transform(y_pred) 
y_target = scaler.inverse_transform(y_train) 

# rmse
mse = np.square(np.subtract(y_target,y_pred)).mean()
baseline_metric = math.sqrt(mse)

X_train = df_train.iloc[:, [6,15,11,17,12,13,7]] 
y_train = df_train.iloc[:, [25]]

#Select the testing set: SC1 or SC2
X_test = df_test.iloc[:, [6,15,11,17,12,13,7]] 
y_test = df_test.iloc[:, [25]]


X = df_train.iloc[:, [6,15,11,17,12,13,7]] # 12,5,7,10,4,15
y_train = df_train.iloc[:, [25]]

num_iterations=10


scores={c:[] for c in X.columns}

for c in X.columns:
    
    X_train = df_train.iloc[:, [6,15,11,17,12,13,7]]
    #print(c)
    X1=X_train.copy(deep=True)
    for _ in range(num_iterations):
        
       temp=X1[c].tolist()
       random.shuffle(temp)
       X1[c]=temp
       
       y_train = df_train.iloc[:, [25]]
               
       scaler = StandardScaler()
       X_train = scaler.fit_transform(X1)
       X_train, y_train = np.array(X_train), np.array(y_train)
        
       #normalize output
       y_train = y_train.reshape(-1, 1)
       y_train = scaler.fit_transform(y_train)
        
       y_pred_train=gtb_regressor.predict(X_train)
      
       # desnormalization
       y_pred = y_pred_train.reshape(-1,1)
       y_pred = scaler.inverse_transform(y_pred) 
       y_target = scaler.inverse_transform(y_train) 
        
       #rmse
       mse = np.square(np.subtract(y_target,y_pred)).mean()
       score = math.sqrt(mse)
        
       scores[c].append((baseline_metric-score)*-1)

#%%

"""
Applying Cross-Validation
"""

max_depth = np.arange(2, 13, 1) #np.arange(2, 13, 1) #define maximum depth
    
min_samples_leaf = np.arange(10, 24, 2)  #define minumum samples leaf

max_iter = len(max_depth)*len(min_samples_leaf)  #number of iteractions

iteration = list(range(1,max_iter+1))

num_trees = np.arange(2, 72, 2) #number of trees

p = len(num_trees)*max_iter
zmatrix_cv_gb = np.zeros((p, 5)) #size of the cross-validation matriz

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
        
        model = GradientBoostingRegressor(n_estimators=num_trees[k], learning_rate=0.14, max_depth=max_depth[md], min_samples_leaf=min_samples_leaf[ml], alpha=0.9, random_state=42, loss='absolute_error', max_features='auto')
        
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
        zmatrix_cv_gb[count][0] = num_trees[k]
        zmatrix_cv_gb[count][1] = max_depth[md]
        zmatrix_cv_gb[count][2] = min_samples_leaf[ml]
        zmatrix_cv_gb[count][3] = RMSE_tot_train
        zmatrix_cv_gb[count][4] = RMSE_tot_val
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
    

#%%

#Marginal contribution

RMSE_delta=[10.87,1.63,2.13,0.85,0.37,0.35,1.09,0.38,0.08,-0.04,0.33,0.12]
IAS=[0,0,-0.12,-0.01,0,0.01,-0.17,-0.02,0.18,0.01,0.01,0]
MEC=[1,-0.23,-0.86,0.31,-0.01,0.05,-0.13,-0.07,-0.31,-0.35,0.03,-0.2]
x=[1,2,3,4,5,6,7,8,9,10,11,12]


plt.plot(x,RMSE_delta,linewidth=2.2,label='$\Delta RMSE_p$ [dB]')
plt.plot(x,IAS,'--',linewidth=2.2,label='$\Delta IAS_p$')
plt.plot(x,MEC,'-.',linewidth=2.2,label='$\Delta MEC_p$')

plt.grid(ls='--')
plt.legend()
plt.xlabel('Predictor subsets',fontsize=12)
plt.ylabel('Marginal contribution',fontsize=12)
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)

plt.xlim(1, 12)

plt.savefig('MC_V2I.eps',format='eps',dpi=1200)
plt.show()
plt.close()


