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
Read Data
"""
#path = r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\Outdoor_data_vegetationdepth.csv"
#path = r"F:\Medidas TEST\Medidas_Outdoor_Tx-ground.csv"

path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\Outdoor_data_vegetationdepth_qgis_final.csv"
df = pd.read_csv(path)
df.head()

"""
Create Input and Output Data
"""
X = df.iloc[:, [1]] #1,0,5,2
y = df.iloc[:, [3]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=90)


df_train = X_train
df_test = X_test
df_train_y = y_train
df_test_y = y_test


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


gtb_regressor = GradientBoostingRegressor(n_estimators=161, 
                                          learning_rate=0.1,
                                          max_depth=3, 
                                          min_samples_leaf=2, 
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


MSE = np.square(np.subtract(y_target_desn,y_pred_desn)).mean() #RMSE
RMSE_train = math.sqrt(MSE)
print(RMSE_train)

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


import pymint #version: 0.2.6
import pickle

#name of predictors

#'building_depth_3d','d_2d','freq','avg_ground_heigth','delta_h_tx_rx','vegetation_depth_3d','avg_building_heigth'

#'building_depth_3d','d_2d','freq','avg_ground_heigth','delta_h_tx_rx','vegetation_depth_3d','avg_vegetation_heigth','avg_building_heigth','n_vegetation','avg_diffracted_comp','n_building'

features_list=['dist'#,'freq'#,'veg_depth'#,'heigth'
               ]
   
X_train_df = pd.DataFrame(X_train, columns = features_list)
   
gtb_regressor = GradientBoostingRegressor(n_estimators=161, learning_rate=0.1, max_depth=3, min_samples_leaf=2, alpha=0.9, random_state=42, loss='ls', max_features='auto')
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

# All predictors
#predictor dist
ale_dist_value_s4 = ale_model.dist__GTB__ale.to_masked_array()  
ale_dist_value_s4= ale_dist_value_s4.ravel() #to reduce 1 dim
ale_dist_bin_s4= ale_model.dist__bin_values.to_masked_array()

#predictor freq
ale_freq_value_s4 = ale_model.freq__GTB__ale.to_masked_array()  
ale_freq_value_s4 = ale_freq_value_s4.ravel() #to reduce 1 dim
ale_freq_bin_s4= ale_model.freq__bin_values.to_masked_array()

#predictor vdepth
ale_veg_depth_value_s4 = ale_model.veg_depth__GTB__ale.to_masked_array()  
ale_veg_depth_value_s4 = ale_veg_depth_value_s4.ravel() #to reduce 1 dim
ale_veg_depth_bin_s4 = ale_model.veg_depth__bin_values.to_masked_array()

#predictor heigth
ale_heigth_value_s4 = ale_model.heigth__GTB__ale.to_masked_array()  
ale_heigth_value_s4 = ale_heigth_value_s4.ravel() #to reduce 1 dim
ale_heigth_bin_s4 = ale_model.heigth__bin_values.to_masked_array()

#%%

"3 predictors"
#predictor dist
ale_dist_value_s3 = ale_model.dist__GTB__ale.to_masked_array()  
ale_dist_value_s3= ale_dist_value_s3.ravel() #to reduce 1 dim
ale_dist_bin_s3= ale_model.dist__bin_values.to_masked_array()

#predictor freq
ale_freq_value_s3 = ale_model.freq__GTB__ale.to_masked_array()  
ale_freq_value_s3 = ale_freq_value_s3.ravel() #to reduce 1 dim
ale_freq_bin_s3= ale_model.freq__bin_values.to_masked_array()

#predictor vdepth
ale_veg_depth_value_s3 = ale_model.veg_depth__GTB__ale.to_masked_array()  
ale_veg_depth_value_s3 = ale_veg_depth_value_s3.ravel() #to reduce 1 dim
ale_veg_depth_bin_s3 = ale_model.veg_depth__bin_values.to_masked_array()

#%%

"2 predictors"
#predictor dist
ale_dist_value_s2 = ale_model.dist__GTB__ale.to_masked_array()  
ale_dist_value_s2= ale_dist_value_s2.ravel() #to reduce 1 dim
ale_dist_bin_s2= ale_model.dist__bin_values.to_masked_array()

#predictor freq
ale_freq_value_s2 = ale_model.freq__GTB__ale.to_masked_array()  
ale_freq_value_s2 = ale_freq_value_s2.ravel() #to reduce 1 dim
ale_freq_bin_s2= ale_model.freq__bin_values.to_masked_array()

#%%

"1 predictors"
#predictor dist
ale_dist_value_s1 = ale_model.dist__GTB__ale.to_masked_array()  
ale_dist_value_s1= ale_dist_value_s1.ravel() #to reduce 1 dim
ale_dist_bin_s1= ale_model.dist__bin_values.to_masked_array()



#%%

import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15,3))
spec = gridspec.GridSpec(ncols=4, nrows=1) # 6 columns evenly divides both 2 & 3

ax1 = fig.add_subplot(spec[0,0]) # row 0 with axes spanning 2 cols on evens
ax2 = fig.add_subplot(spec[0,1]) # row 0 with axes spanning 2 cols on evens
ax3 = fig.add_subplot(spec[0,2]) # row 0 with axes spanning 2 cols on evens
ax4 = fig.add_subplot(spec[0,3]) # row 0 with axes spanning 2 cols on evens
#ax5 = fig.add_subplot(spec[4,0]) # row 0 with axes spanning 2 cols on evens


#d
ax1.plot(ale_dist_bin_s1,ale_dist_value_s1,"+-", label ='1',lw=4,color='blue')
ax1.plot(ale_dist_bin_s2,ale_dist_value_s2,"-.", label ='2',lw=4,color='red')
ax1.plot(ale_dist_bin_s3,ale_dist_value_s3,"--", label ='3',lw=4,color='green')
ax1.plot(ale_dist_bin_s4,ale_dist_value_s4, label ='4',lw=3,color='orange')
ax1.grid(ls='--')
ax1.set_xlabel('$d$ (m)',fontsize=18)
ax1.set_title('(a)', fontsize=18)
ax1.xaxis.set_tick_params(labelsize=16)
ax1.yaxis.set_tick_params(labelsize=16)
ax1.legend(fancybox=False,shadow=False,fontsize="13")
x_labels1 = ['','54','108','162','216','271']
ax1.set_xticklabels(x_labels1)
    

#f
ax2.plot(ale_freq_bin_s2,ale_freq_value_s2,"-.", label ='2',lw=4,color='red')
ax2.plot(ale_freq_bin_s3,ale_freq_value_s3,"--", label ='3',lw=4,color='green')
ax2.plot(ale_freq_bin_s4,ale_freq_value_s4, label ='4',lw=3,color='orange')
ax2.grid(ls='--')
ax2.set_xlabel('$f$ (GHz)',fontsize=18)
ax1.set_title('(b)', fontsize=18)
ax2.xaxis.set_tick_params(labelsize=16)
ax2.yaxis.set_tick_params(labelsize=16)
ax2.legend(fancybox=False,shadow=False,fontsize="13")
x_labels2 = ['27','30','33','36','40']
ax2.set_xticklabels(x_labels2)


#veg_depth
ax3.plot(ale_veg_depth_bin_s3,ale_veg_depth_value_s3,"--", label ='3',lw=4,color='green')
ax3.plot(ale_veg_depth_bin_s4,ale_veg_depth_value_s4, label ='4',lw=4,color='orange')
ax3.grid(ls='--')
ax3.set_xlabel('$v_{depth}$ (m)',fontsize=18)
ax3.set_title('(c)', fontsize=18)
ax3.xaxis.set_tick_params(labelsize=16)
ax3.yaxis.set_tick_params(labelsize=16)
ax3.legend(fancybox=False,shadow=False,fontsize="13")
x_labels3 = ['0','1.4','2.7','4.06','5.4']
ax3.set_xticklabels(x_labels3)

#heigth
ax4.plot(ale_heigth_bin_s4,ale_heigth_value_s4, label ='4',lw=4,color='orange')
ax4.grid(ls='--')
ax4.set_xlabel('$\Delta_h$ (m)',fontsize=18)
ax4.set_title('(d)', fontsize=18)
ax4.xaxis.set_tick_params(labelsize=16)
ax4.yaxis.set_tick_params(labelsize=16)
ax4.legend(fancybox=False,shadow=False,fontsize="13")
x_labels4 = ['15','24.5','34.0','43.5','53.0']
ax4.set_xticklabels(x_labels4)


plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.48,
                    hspace=0.5
                    )

plt.savefig('ALE-GTB-Outdoor.eps',format='eps',dpi=600)
plt.show()
plt.close()

#%%

#2D effects
#interaction 2D-ALE
import itertools

#Interaction effect between two predictors
important_vars = features_list
important_vars_2d = list(itertools.combinations(important_vars,r=2))

ale_2d_ds = explainer_gtb.ale(features=important_vars_2d, n_bootstrap=1, subsample=1.0, n_jobs=6, n_bins=30)
ale_int_2d = explainer_gtb.ale_variance(ale=ale_2d_ds, features=important_vars_2d, interaction=True)

int_2d_s5 = ale_int_2d.ale_variance_interactions_scores__GTB
int_2d_s5 = int_2d_s5.to_masked_array()
int_2d_s5 = int_2d_s5.tolist()

#%%
int_2d_s5_p1 = int_2d_s5[0][0]

#%%
int_2d_s5_p2 = int_2d_s5[1][0]
int_2d_s5_p3 = int_2d_s5[2][0]
int_2d_s5_p4 = int_2d_s5[3][0]
int_2d_s5_p5 = int_2d_s5[4][0]
int_2d_s5_p6 = int_2d_s5[5][0]
int_2d_s5_p7 = int_2d_s5[6][0]
int_2d_s5_p8 = int_2d_s5[7][0]
int_2d_s5_p9 = int_2d_s5[8][0]
int_2d_s5_p10 = int_2d_s5[9][0]


#%%

"PFI value"
X_train = df_train.iloc[:, [0,1,2,3]] # 12,5,7,10,4,15
y_train = df_train_y

#Select the testing set: SC1 or SC2
X_test = df_test.iloc[:, [0,1,2,3]] # 12,5,7,10,412,5,7,10,4,15,15
y_test = df_test_y


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


X_train = df_train.iloc[:, [0,1,2,3]] # 12,5,7,10,4,15
y_train = df_train_y

#Select the testing set: SC1 or SC2
X_test = df_test.iloc[:, [0,1,2,3]] # 12,5,7,10,412,5,7,10,4,15,15
y_test = df_test_y


X = df_train.iloc[:, [0,1,2,3]] # 12,5,7,10,4,15
y_train = df_train_y

num_iterations=10

scores={c:[] for c in X.columns}

for c in X.columns:
    
    X_train = df_train.iloc[:, [0,1,2,3]] # 12,5,7,10,4,15
    #print(c)
    X1=X_train.copy(deep=True)
    for _ in range(num_iterations):
        
       temp=X1[c].tolist()
       random.shuffle(temp)
       X1[c]=temp
       
       #X = df_train.iloc[:, [12,5,7,10,4,15]] # 12,5,7,10,4,15
       y_train = df_train_y
               
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

#%%
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

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(15,4))
spec = gridspec.GridSpec(ncols=4, nrows=1) # 6 columns evenly divides both 2 & 3

ax1 = fig.add_subplot(spec[0,0]) # row 0 with axes spanning 2 cols on evens
ax2 = fig.add_subplot(spec[0,1]) # row 0 with axes spanning 2 cols on evens
ax3 = fig.add_subplot(spec[0,2]) # row 0 with axes spanning 2 cols on evens
ax4 = fig.add_subplot(spec[0,3]) # row 0 with axes spanning 2 cols on evens
#ax5 = fig.add_subplot(spec[4,0]) # row 0 with axes spanning 2 cols on evens

path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\Plot - ALE - Outdoor\ALE-d.csv"
df = pd.read_csv(path)

#d
ax1.plot(df['ale_dist_bin_s1'],df['ale_dist_value_s1'],"+-", label ='1',lw=4,color='blue')
ax1.plot(df['ale_dist_bin_s2'],df['ale_dist_value_s2'],"-.", label ='2',lw=4,color='red')
ax1.plot(df['ale_dist_bin_s3'],df['ale_dist_value_s3'],"--", label ='3',lw=4,color='green')
ax1.plot(df['ale_dist_bin_s4'],df['ale_dist_value_s4'], label ='4',lw=4,color='orange')
ax1.set_xlabel('$d$ (m)',fontsize=20)
ax1.set_title('(a)', fontsize=20)
ax1.grid(ls='--')
ax1.xaxis.set_tick_params(labelsize=20)
ax1.yaxis.set_tick_params(labelsize=20)
ax1.legend(fancybox=False,shadow=False,fontsize="16")
x_labels1 = ['','54','108','162','216','271']
ax1.set_xticklabels(x_labels1)


path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\Plot - ALE - Outdoor\ALE-f.csv"
df = pd.read_csv(path)
#f
ax2.plot(df['ale_freq_bin_s2'],df['ale_freq_value_s2'],"-.", label ='2',lw=4,color='red')
ax2.plot(df['ale_freq_bin_s3'],df['ale_freq_value_s3'],"--", label ='3',lw=4,color='green')
ax2.plot(df['ale_freq_bin_s4'],df['ale_freq_value_s4'], label ='4',lw=4,color='orange')
ax2.set_xlabel('$f$ (GHz)',fontsize=20)
ax2.set_title('(b)', fontsize=20)
ax2.grid(ls='--')
ax2.xaxis.set_tick_params(labelsize=20)
ax2.yaxis.set_tick_params(labelsize=20)
ax2.legend(fancybox=False,shadow=False,fontsize="16")
x_labels2 = ['27','30','33','36','40']
ax2.set_xticklabels(x_labels2)


path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\Plot - ALE - Outdoor\ALE-vd.csv"
df = pd.read_csv(path)

#veg_depth
ax3.plot(df['ale_veg_depth_bin_s3'],df['ale_veg_depth_value_s3'],"--", label ='3',lw=4,color='green')
ax3.plot(df['ale_veg_depth_bin_s4'],df['ale_veg_depth_value_s4'], label ='4',lw=4,color='orange')
ax3.set_xlabel('$v_{depth}$ (m)',fontsize=20)
ax3.set_title('(c)', fontsize=20)
ax3.xaxis.set_tick_params(labelsize=20)
ax3.yaxis.set_tick_params(labelsize=20)
ax3.legend(fancybox=False,shadow=False,fontsize="16")
ax3.grid(ls='--')
x_labels3 = ['0','1.4','2.7','4.06','5.4']
ax3.set_xticklabels(x_labels3)

#heigth
path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\Plot - ALE - Outdoor\ALE-ah.csv"
df = pd.read_csv(path)

ax4.plot(df['ale_heigth_bin_s4'],df['ale_heigth_value_s4'], label ='4',lw=4,color='orange')
ax4.set_xlabel('$\Delta_h$ (m)',fontsize=20)
ax4.set_title('(d)', fontsize=20)
ax4.xaxis.set_tick_params(labelsize=20)
ax4.yaxis.set_tick_params(labelsize=20)
ax4.legend(fancybox=False,shadow=False,fontsize="13")
ax4.grid(ls='--')
x_labels4 = ['15','24.5','34.0','43.5','53.0']
ax4.set_xticklabels(x_labels4)

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.35,
                    hspace=0.7
                    )
fig.tight_layout()
plt.savefig('ALE-GTB-Outdoor.eps',format='eps',dpi=600)

plt.show()
plt.close()

#%%


import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12,4))
spec = gridspec.GridSpec(ncols=2, nrows=1) 

ax1 = fig.add_subplot(spec[0,0]) 
ax2 = fig.add_subplot(spec[0,1]) 

path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\R2_plot_gen_outdoor\R2_final_model_outdoor_2.csv"
df = pd.read_csv(path)
df.head()

#Plot
p1 = max(max(df["y_target"]), max(df["y_target"]))
p2 = min(min(df["y_target"]), min(df["y_target"]))
ax1.plot([p1, p2], [p1, p2], '--',linewidth=1,label= 'Best fit',c='navy')
ax1.scatter(df["y_target"],df["pl_pred_ann"],s=75, label= 'ANN, $R^2 = 0.78$',c='olivedrab',marker='+')
ax1.scatter(df["y_target"],df["pl_pred_svr"],s=55, label= 'SVR, $R^2 = 0.79$',c='darksalmon',marker='D',linewidths=0)
ax1.scatter(df["y_target"],df["pl_pred_rf"],s=55, label= 'RF,  $R^2 = 0.85$',c='turquoise')
ax1.scatter(df["y_target"],df["pl_pred_gtb"],s=55,label= 'GTB, $R^2 = 0.88$',c='crimson',marker='*')
ax1.set_xlabel('Measured path loss [dB]',fontsize=14)
ax1.set_ylabel('Predicted path loss [dB]',fontsize=14)
ax1.set_title('(a) ML based models', fontsize=14)
ax1.grid(ls='--')
ax1.legend()
ax1.set_ylim(102,125)
ax1.set_xlim(102,125)
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)


#Plot
p1 = max(max(df["y_target"]), max(df["y_target"]))
p2 = min(min(df["y_target"]), min(df["y_target"]))
ax2.plot([p1, p2], [p1, p2], '--',linewidth=1,label= 'Best fit',c='navy')
ax2.scatter(df["y_target"],df["pl_pred_abg"],s=75, label= 'ABG, $R^2 = 0.61$',c='olivedrab',marker='+')
ax2.scatter(df["y_target"],df["pl_pred_abg_p"],s=55, label= 'ABG$\Delta_h,v_{depth}$, $R^2 = 0.66$',c='darksalmon',marker='D',linewidths=0)
ax2.scatter(df["y_target"],df["pl_pred_cif"],s=55, label= 'CIF,  $R^2 = 0.54$',c='turquoise')
ax2.scatter(df["y_target"],df["pl_pred_cif_p"],s=55,label= 'CIF$\Delta_h,v_{depth}$, $R^2 = 0.65$',c='crimson',marker='*')
ax2.set_xlabel('Measured path loss [dB]',fontsize=14)
ax2.set_ylabel('Predicted path loss [dB]',fontsize=14)
ax2.set_title('(b) Empirical based models', fontsize=14)
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)

ax2.set_ylim(102,125)
ax2.set_xlim(102,125)
ax2.grid(ls='--')
ax2.legend()

fig.tight_layout()
plt.savefig('r2_outdoor.eps',format='eps',dpi=1200)
plt.show()
plt.close()