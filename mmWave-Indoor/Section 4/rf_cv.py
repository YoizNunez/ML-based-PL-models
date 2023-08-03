# -*- coding: utf-8 -*-

#immporting the libraries
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
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


def rf_model(df):

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
    
    max_depth = list(range(3,7)) #define maximum depth
    
    min_samples_leaf = list(range(1,4)) #define minumum samples leaf
    
    max_iter = len(max_depth)*len(min_samples_leaf) #number of iteractions
    
    iteration = list(range(1,max_iter+1))
    
    num_trees = np.arange(8, 256,17) # Number of trees
    
    p = len(num_trees)*max_iter
    matrix_cv_rf = np.zeros((p, 4)) # Size of the CV matriz
    
    cv = KFold(n_splits=5,shuffle=True,random_state=42)
    
    
    count = 0
    md = 0
    ml = 0
    c = 1
    a = 0
    b = 0
    x=0
    
    for i in range(max_iter):
        
        for k in range(len(num_trees)):
            
            rfmodel = RandomForestRegressor(n_estimators=num_trees[k], max_depth=max_depth[md], min_samples_split=2, min_samples_leaf=min_samples_leaf[ml], max_features = 'auto', random_state=1)
            
            RMSE_tot = 0
            
            for train, test in cv.split(X_train, y_train):
                rfmodel.fit(X_train[train], np.ravel(y_train[train]))
                y_pred = rfmodel.predict(X_train[test])
                
                #desnormalization
                y_pred = y_pred.reshape(-1,1)
                y_pred_desn = scaler.inverse_transform(y_pred)
                y_target_desn = scaler.inverse_transform(y_train[test])
                
                #RMSE
                MSE = np.square(np.subtract(y_target_desn,y_pred_desn)).mean()
                RMSE_desn = math.sqrt(MSE)
                RMSE_tot +=RMSE_desn
                
            RMSE_tot = RMSE_tot/5
           
            matrix_cv_rf[count][0] = num_trees[k]
            matrix_cv_rf[count][1] = max_depth[md]
            matrix_cv_rf[count][2] = min_samples_leaf[ml]
            matrix_cv_rf[count][3] = RMSE_tot
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
    Hyperparameter selection for the RF Model
    """
    p_idx = matrix_cv_rf[:,3].argmin() #selection of the minimum value of RMSE validation
    num_tree_rf= int(matrix_cv_rf[p_idx][0]) #number of trees
    max_depth_rf=int(matrix_cv_rf[p_idx][1]) #maximum depth
    min_samples_leaf_rf=int(matrix_cv_rf[p_idx][2]) #minimum samples leaf
    
    data = [['CV RMSE [dB] - validation', matrix_cv_rf[p_idx][3]],
    ['Number of trees',num_tree_rf],
    ['Maximum depth', max_depth_rf],
    ['minimum samples leaf',min_samples_leaf_rf]]
    
    print(tabulate(data, headers=["Hyperparameters RF Model","Value",]))
    
    start_time = time.time() #training time in second
    rfmodel = RandomForestRegressor(n_estimators=num_tree_rf, max_depth=max_depth_rf, max_features = 'auto', min_samples_split=2, min_samples_leaf=min_samples_leaf_rf, random_state=0)
    rfmodel.fit(X_train, np.ravel(y_train))
    time_rf= time.time() - start_time
    
    #Saving model to know the size in KB of the saved file
    pickle.dump(rfmodel, open('rfmodel.pkl','wb'))
    file= r"C:\Users\Yoiz Nuñez\rfmodel.pkl" #set the local path where the file .pkl is saved
    file_stats=os.stat(file)
    file_KB=file_stats.st_size/(1024)
    
    y_pred_train = rfmodel.predict(X_train)
    y_pred_test = rfmodel.predict(X_test)
    
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
    
    
    rf_results={"train_rf":[y_pred_desn_train,y_target_desn_train,RMSE_train,MAPE_train,SD_train,R2_test],
              "test_rf":[y_pred_desn_test,y_target_desn_test,RMSE_test,MAPE_test,SD_test,R2_test]}
    
    return rf_results, num_tree_rf, time_rf, file_KB