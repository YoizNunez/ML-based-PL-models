# -*- coding: utf-8 -*-

#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import math
import tabulate
from tabulate import tabulate
import pickle
import time
import os

from util_tools import performance_indicators

import warnings
warnings.filterwarnings("ignore")


def gtb_model(df):
    
    """
    Create input and output Data, 80% for training, 20% for testing
    """
    X = df.iloc[:,[0,1,2,3]]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #normalization of numerical inputs
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    #concatenate with the one-hot encoding categorical input
    X = df.iloc[:,[4]]
    X_nw_train, X_nw_test = train_test_split(X, test_size=0.2, random_state=42)
    X_train= np.concatenate([X_train,X_nw_train],axis=1)
    X_test= np.concatenate([X_test,X_nw_test],axis=1)
    
    #normalize output
    y_train = y_train.reshape(-1, 1)
    y_train = scaler.fit_transform(y_train)
    y_test = y_test.reshape(-1, 1)
    y_test = scaler.transform(y_test)
    
    #convert output variable to float
    y_train, y_test = y_train.astype(float), y_test.astype(float),
        
    """
    Applying cross-validation using a grid search method
    """
    
    max_depth = list(range(2,6)) #define maximum depth
    
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
            
            model = GradientBoostingRegressor(n_estimators=num_trees[k], learning_rate=0.1, max_depth=max_depth[md], min_samples_leaf=min_samples_leaf[ml], alpha=0.9, random_state=0, loss='ls', max_features='auto')
            
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
    
    
    """
    Hyperparameter selection for the GTB Model
    """
    p_idx = matrix_cv_gb[:,4].argmin() #selection of the minimum value of RMSE validation
    num_tree_gtb= int(matrix_cv_gb[p_idx][0]) #number of trees
    max_depth_gtb=int(matrix_cv_gb[p_idx][1]) #maximum depth
    min_samples_leaf_gtb=int(matrix_cv_gb[p_idx][2]) #minimum samples leaf
    
    data = [['CV RMSE [dB] - validation set', matrix_cv_gb[p_idx][4]],
    ['Number of trees',num_tree_gtb],
    ['Maximum depth', max_depth_gtb],
    ['minimum samples leaf',min_samples_leaf_gtb]]
    
    print(tabulate(data, headers=["Hyperparameters GTB model","Value",]))
    
    """
    Training of the GTB model using the hyperparameter setting calculated in the grid search method
    """
    start_time = time.time() #training time in second
    gbmodel = GradientBoostingRegressor(n_estimators=num_tree_gtb,learning_rate=0.1, max_depth=max_depth_gtb, min_samples_leaf=min_samples_leaf_gtb, alpha=0.1, random_state=42, loss='ls', max_features=None)
    gbmodel.fit(X_train, np.ravel(y_train))
    time_gtb= time.time() - start_time
    
    #saving model to know the size in KB of the saved file
    pickle.dump(gbmodel, open('gtbmodel.pkl','wb')) #file directory where the model is saved
    file= r"C:\Users\Yoiz Nuñez\gtbmodel.pkl" #set the local path where the file .pkl is saved
    file_stats=os.stat(file)
    file_KB=file_stats.st_size/(1024)

    y_pred_train = gbmodel.predict(X_train) 
    y_pred_test = gbmodel.predict(X_test)
       
    """
    Performance indicators for the training set
    """
    #desnormalization to calculate the performance indicators RMSE, MAPE, SD e R2 for the results in Table 4
    y_pred_desn = y_pred_train.reshape(-1,1)
    y_pred_desn_train= scaler.inverse_transform(y_pred_desn)
    y_target_desn_train = scaler.inverse_transform(y_train)
    
    #performance indicators results
    RMSE_train, MAPE_train, SD_train, R2_train= performance_indicators(y_target_desn_train,y_pred_desn_train)
    
    """
    Performance indicators for the testing set
    """
    #desnormalization to calculate the performance indicators RMSE, MAPE, SD e R2 for the results in Table 4
    y_pred_desn = y_pred_test.reshape(-1,1)
    y_pred_desn_test= scaler.inverse_transform(y_pred_desn)
    y_target_desn_test = scaler.inverse_transform(y_test)
    
    #performance indicators results
    RMSE_test, MAPE_test, SD_test, R2_test= performance_indicators(y_target_desn_test,y_pred_desn_test)
    

    gtb_results={"train_gtb":[y_pred_desn_train,y_target_desn_train,RMSE_train,MAPE_train,SD_train,R2_test],
              "test_gtb":[y_pred_desn_test,y_target_desn_test,RMSE_test,MAPE_test,SD_test,R2_test]}
    
    return gtb_results, num_tree_gtb, time_gtb, file_KB
















