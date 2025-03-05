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
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

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
path = r"Coord_Tx_Rx_V2V_Suburban.csv"
df = pd.read_csv(path)
df.head()

n=1000
df_train= df.iloc[:n]

n=374
df_test= df.tail(n)

#%%

"""
Create Input and Output Data
"""

X_train = df_train.iloc[:, [13,14,16,17]] #with vegetation depth -->5 
y_train = df_train.iloc[:, [8]]

#Select the testing set: SC1 or SC2
X_test = df_test.iloc[:, [13,14,16,17]] #with vegetation depth -->5 
y_test = df_test.iloc[:, [8]]

#%%
d= df.iloc[:, [12]]
pl= df.iloc[:, [8]]

#Plot
plt.scatter(d,pl,s=20)
plt.grid(ls='--')
plt.legend()
plt.xlabel('d [m]',fontsize=12)
plt.ylabel('PL [dB]',fontsize=12)
plt.savefig('d_PL_V2V.eps',format='eps',dpi=1200)
plt.show()
plt.close()

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

gtb_regressor =  RandomForestRegressor(n_estimators=12, max_depth=4, min_samples_leaf=10, criterion='absolute_error', random_state=42)

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

#%%
import pymint #version: 0.2.6
import pickle

features_list=['d','v']
   
X_train_df = pd.DataFrame(X_train, columns = features_list)
   
    
gtb_regressor =  RandomForestRegressor(n_estimators=14, max_depth=3, min_samples_leaf=10,criterion='squared_error', random_state=42)

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

"""
Applying Cross-Validation
"""

max_depth = np.arange(2, 5, 1) #np.arange(2, 13, 1) #define maximum depth
    
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
        
        model = RandomForestRegressor(n_estimators=num_trees[k], max_depth=max_depth[md], min_samples_leaf=min_samples_leaf[ml],criterion='squared_error', random_state=42)
    
        
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

cv = KFold(n_splits=5,shuffle=True,random_state=42)

X_train=X_train.to_numpy()
y_train=y_train.to_numpy()

a=1

#iterations
for train, test in cv.split(X_train, y_train):
    
    if a==1:
        X_train_subset1=X_train[train]
        y_train_subset1=y_train[train]
        X_val_subset1=X_train[test]
        y_val_subset1=y_train[test]

    if a==2:
        X_train_subset2=X_train[train]
        y_train_subset2=y_train[train]
        X_val_subset2=X_train[test]
        y_val_subset2=y_train[test]

    if a==3:
        X_train_subset3=X_train[train]
        y_train_subset3=y_train[train]
        X_val_subset3=X_train[test]
        y_val_subset3=y_train[test]


    if a==4:
        X_train_subset4=X_train[train]
        y_train_subset4=y_train[train]
        X_val_subset4=X_train[test]
        y_val_subset4=y_train[test]
        
    if a==5:
        X_train_subset5=X_train[train]
        y_train_subset5=y_train[train]
        X_val_subset5=X_train[test]
        y_val_subset5=y_train[test]

    a+=1

#%%

#CIF Model
#Number of frequencies repetitions

X_train_subset1=pd.DataFrame(X_train_subset1,columns=['f','d','h','vd'])
X_train_subset2=pd.DataFrame(X_train_subset2,columns=['f','d','h','vd'])
X_train_subset3=pd.DataFrame(X_train_subset3,columns=['f','d','h','vd'])
X_train_subset4=pd.DataFrame(X_train_subset4,columns=['f','d','h','vd'])
X_train_subset5=pd.DataFrame(X_train_subset5,columns=['f','d','h','vd'])

df2 = X_train.pivot_table(index = ['freq'], aggfunc ='size') 
df2=df2.to_numpy()

df1 = X_train['freq'].drop_duplicates()
df1 = df1.sort_values(ascending=True)
df1=df1.to_numpy()

fk_NK=np.arange(0,len(df2))

#%%
for i in range(len(df2)):
    
    fk_NK[i]=df1[i]*df2[i] #f*Nk
    
    
total_Nk=np.sum(df2)  #Nk
total_fk_Nk=np.sum(fk_NK)  

fo=total_fk_Nk/total_Nk

print(fo)
    
#%%
import math

path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\Outdoor Results\CV EM\Test.csv"
df = pd.read_csv(path)
df.head()

f=df['f']
d=df['d']
h=df['h']
vd=df['vd']
fslp=df['fslp']
pl_measured = df['pl'].to_numpy()


#ABG
a=2.29793349331932
b=63.9109590549251
g=0.21247699594073
k=3.97114642573868
m=0.166650690941781


pl_pred=np.arange(0,len(df['f']),dtype=np.float64)

for i in range(len(df['f'])):
    
    pl_pred[i] = 10*a*math.log10(d[i]) + b + 10*g*math.log10(f[i]) + m*vd[i] +k*math.log10(h[i]/d[i])


#%%
MSE = np.square(np.subtract(pl_measured,pl_pred)).mean() #RMSE
RMSE = math.sqrt(MSE)
print(RMSE)

MAPE = np.mean(np.abs((pl_measured - pl_pred)/pl_measured))*100 #MAPE
print(MAPE)

R2= r2_score(pl_measured,pl_pred) #R2
print(R2)

n = len(pl_pred)
sum_model=0

abs_dif = np.abs(pl_measured,pl_pred)
mean_model = np.mean(abs_dif)

for x in abs_dif:
    t = (x - mean_model) ** 2
    sum_model += t 

SD = math.sqrt(sum_model/(n)) #SD
print(SD)


#%%
import math

path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\Outdoor Results\CV EM\Test.csv"
df = pd.read_csv(path)
df.head()


f=df['f']
d=df['d']
h=df['h']
vd=df['vd']
fslp=df['fslp']
pl_measured = df['pl'].to_numpy()


#CIF
n=0.536869175469791
b=-0.730407785078322
m=0.137545537755334
k=5.10558978838471
fo=33.51

pl_pred=np.arange(0,len(df['f']),dtype=np.float64)

for i in range(len(df['f'])):
    
    pl_pred[i] = fslp[i] + 10*n*math.log10(d[i])+10*n*b*((f[i]-fo)/fo)*math.log10(d[i]) + m*vd[i] + k*math.log10(h[i]/d[i])

#%%
MSE = np.square(np.subtract(pl_measured,pl_pred)).mean() #RMSE
RMSE = math.sqrt(MSE)
print(RMSE)

MAPE = np.mean(np.abs((pl_measured - pl_pred)/pl_measured))*100 #MAPE
print(MAPE)

R2= r2_score(pl_measured,pl_pred) #R2
print(R2)

n = len(pl_pred)
sum_model=0

abs_dif = np.abs(pl_measured,pl_pred)
mean_model = np.mean(abs_dif)

for x in abs_dif:
    t = (x - mean_model) ** 2
    sum_model += t 

SD = math.sqrt(sum_model/(n)) #SD
print(SD)

#%%

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

fig, ax = plt.subplots(figsize = (8,7))
ax.scatter(df_train_SC2_750['Long'], df_train_SC2_750['Lat'], c='black', s=20)
#ax.scatter(df_test_SC2_750['Long'], df_test_SC2_750['Lat'], c='blue', s=20)


fig, ax = plt.subplots(figsize = (8,7))
ax.scatter(df_train_SC2_2500['Long'], df_train_SC2_2500['Lat'], c='black', s=20)
#ax.scatter(df_test_SC2_2500['Long'], df_test_SC2_2500['Lat'], c='green', s=20)


fig, ax = plt.subplots(figsize = (8,7))
ax.scatter(df_train_SC2_3500['Long'], df_train_SC2_3500['Lat'], c='black', s=20)
#ax.scatter(df_test_SC2_3500['Long'], df_test_SC2_3500['Lat'], c='orange', s=20)


#%%


path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\V2I\SC1_3500_Train_1.csv"
df_3500_train = pd.read_csv(path)
df_3500_train.head()

path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\V2I\SC1_3500_Test_1.csv"
df_3500_test = pd.read_csv(path)
df_3500_test.head()

fig, ax = plt.subplots(figsize = (8,7))
ax.scatter(df_3500_test['Long'], df_3500_test['Lat'], c='blue', s=20)
ax.scatter(df_3500_train['Long'], df_3500_train['Lat'], c='black', s=20)

#%%

ruh_m = plt.imread('C:/Users/Yoiz Nuñez/Documents/DOUTORADO 2023/V2I/Measurement_campaign.png')

BBox = (df.Long.min(),df.Long.max(),df.Lat.min(), df.Lat.max())
        
        
fig, ax = plt.subplots(figsize = (8,7))

ax.scatter(df_train.Long, df_train.Lat, zorder=1, alpha= 0.2, c='blue', s=10)
ax.scatter(df_test_SC1.Long, df_test_SC1.Lat, zorder=1, alpha= 0.2, c='red', s=10)
ax.scatter(df_test_SC2.Long, df_test_SC2.Lat, zorder=1, alpha= 0.2, c='magenta', s=10)

ax.set_title('Case #3')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylabel('Latitude')
ax.set_xlabel('Longitude')

ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')

#%%
import pymint #version: 0.2.6
import pickle

#name of predictors

#'building_depth_3d','d_2d','freq','avg_ground_heigth','delta_h_tx_rx','vegetation_depth_3d','avg_building_heigth'

#'building_depth_3d','d_2d','freq','avg_ground_heigth','delta_h_tx_rx','vegetation_depth_3d','avg_vegetation_heigth','avg_building_heigth','n_vegetation','avg_diffracted_comp','n_building'

features_list=['building_depth_3d','d_2d','freq','avg_ground_heigth','delta_h_tx_rx','vegetation_depth_3d','avg_vegetation_heigth','avg_building_heigth','n_vegetation','avg_diffracted_comp','n_building']
   
X_train_df = pd.DataFrame(X_train, columns = features_list)
   
    
#%%
gtb_regressor = GradientBoostingRegressor(n_estimators=60, learning_rate=0.1, max_depth=3, min_samples_leaf=1, alpha=0.9, random_state=0, loss='ls', max_features='auto')
gtb_regressor.fit(X_train, np.ravel(y_train))
        
# saving the model as pickle file to be reading by the library pymint, according its especification.
pickle.dump(gtb_regressor, open('gtbmodel_IML.pkl','wb')) #save the model, i.e., 'mlpmodel_IML_5.pkl' for the subset of 5 predictors
gtb = pickle.load(open('gtbmodel_IML.pkl','rb')) #the file is load from the local path. In my case from C:/Users/Yoiz Nuñez
gtb_model = ('GTB',gtb)

#%%
#ALE
explainer_gtb = pymint.InterpretToolkit(gtb_model,X=X_train_df, y=y_train)
ale_model = explainer_gtb.ale(features=features_list, n_bins=30, n_jobs=1, subsample=1.0, n_bootstrap=1)

MEC = explainer_gtb.main_effect_complexity(ale_model,max_segments=10)

IAS= explainer_gtb.interaction_strength(ale_model)
