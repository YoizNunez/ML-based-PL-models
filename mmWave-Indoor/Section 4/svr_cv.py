# -*- coding: utf-8 -*-

#importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
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

def svr_model(df):
    
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
    C = np.arange(200,2200,100) 
    epsilon= [0.005,0.05,0.1] 
    gamma=[0.1, 0.2, 0.3]
    
    max_iter= len(epsilon)
    
    p = len(C)*max_iter
    matrix_cv_svr = np.zeros((p, 4)) # Size of the CV matrix
    
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
    for i in range(len(epsilon)):
        
        for k in range(len(C)):
            
            model = SVR(kernel='rbf', C=C[k], epsilon=epsilon[i],gamma=0.2)
            
            RMSE_tot = 0
            
            for train, test in cv.split(X_train, y_train):
                model.fit(X_train[train], np.ravel(y_train[train]))
                y_pred = model.predict(X_train[test])
                
                #desnormalization
                y_pred = y_pred.reshape(-1,1)
                y_pred_desn = scaler.inverse_transform(y_pred)
                y_target_desn = scaler.inverse_transform(y_train[test])
                #RMSE
                MSE = np.square(np.subtract(y_target_desn,y_pred_desn)).mean()
                RMSE_desn = math.sqrt(MSE)
                RMSE_tot +=RMSE_desn
                              
            RMSE_tot = RMSE_tot/5
           
            matrix_cv_svr[count][0] = C[k]
            matrix_cv_svr[count][1] = epsilon[i]
            matrix_cv_svr[count][2] = 0.1
            matrix_cv_svr[count][3] = RMSE_tot
            count += 1
            
    """
    Hyperparameter selection for the SVR Model
    """
    p_idx = matrix_cv_svr[:,3].argmin() #selection of the minimum value of RMSE validation
    C= int(matrix_cv_svr[p_idx][0]) #number of trees
    epsilon=matrix_cv_svr[p_idx][1] #maximum depth
    
    data = [['CV RMSE [dB] - validation', matrix_cv_svr[p_idx][3]],
    ['Regularization factor C',C],
    ['epsilon', epsilon]]
    
    print(tabulate(data, headers=["Hyperparameters SVR Model","Value",]))
    
    start_time = time.time() #training time in second
    svrmodel = SVR(kernel='rbf',C=C, epsilon=epsilon,gamma=0.2)
    svrmodel.fit(X_train, np.ravel(y_train))
    time_svr= time.time() - start_time
    
    num_sv=svrmodel.n_support_  #number of support vectors
    
    #Saving model to know the size in KB of the saved file
    pickle.dump(svrmodel, open('svrmodel.pkl','wb'))
    file= r"C:\Users\Yoiz Nuñez\svrmodel.pkl" #set the local path where the file .pkl is saved
    file_stats=os.stat(file)
    file_KB=file_stats.st_size/(1024)
    
    y_pred_train = svrmodel.predict(X_train)
    y_pred_test = svrmodel.predict(X_test)
    
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
    
    svr_results={"train_svr":[y_pred_desn_train,y_target_desn_train,RMSE_train,MAPE_train,SD_train,R2_test],
              "test_svr":[y_pred_desn_test,y_target_desn_test,RMSE_test,MAPE_test,SD_test,R2_test]}
    
    return svr_results, num_sv,time_svr,file_KB
