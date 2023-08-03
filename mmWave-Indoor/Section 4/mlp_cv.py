# -*- coding: utf-8 -*-

#importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
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


def mlp_model(df):
        
    """
    Create input and output Data, 80% for training, 20% for testing
    """
    X = df.iloc[:,[2,3,0]]
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
    
    #concatenate with the one-hot encoding categorical input
    X = df.iloc[:,[1]]
    X_d_train, X_d_test = train_test_split(X, test_size=0.2, random_state=42)
    X_d_train = scaler.fit_transform(X_d_train)
    X_d_test = scaler.transform(X_d_test)
    
    X_train= np.concatenate([X_train,X_d_train],axis=1)
    X_test= np.concatenate([X_test,X_d_test],axis=1)
    
    #normalize output
    y_train = y_train.reshape(-1, 1)
    y_train = scaler.fit_transform(y_train)
    y_test = y_test.reshape(-1, 1)
    y_test = scaler.transform(y_test)
    
    #convert output variable to float
    y_train, y_test = y_train.astype(float), y_test.astype(float),
    
    """
    Applying Cross-Validation
    """
    learning_rate = [0.001,0.01,0.1] #define maximum depth
    
    alpha = [0.001,0.01,0.1] 
    
    max_iter = len(learning_rate)*len(alpha)
    
    iteration = list(range(1,max_iter+1))
    
    num_neurons = np.arange(10,90,4) # Number of neurons
    
    p = len(num_neurons)*max_iter
    matrix_cv_mlp = np.zeros((p, 4)) # Size of the CV matriz
    
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
    
    for i in range(max_iter):
        
        for k in range(len(num_neurons)):
            
            model = MLPRegressor(hidden_layer_sizes=num_neurons[k], 
                                 activation='relu',
                                 solver='lbfgs',
                                 alpha=alpha[md],
                                 learning_rate_init=learning_rate[ml],
                                 momentum=0.9,
                                 random_state=1, 
                                 max_iter=6000,
                                 early_stopping=True)
            
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
           
            matrix_cv_mlp[count][0] = num_neurons[k]
            matrix_cv_mlp[count][1] = learning_rate[md]
            matrix_cv_mlp[count][2] = alpha[ml]
            matrix_cv_mlp[count][3] = RMSE_tot
            count += 1
            c +=1
            
            if c == a + len(num_neurons)+1:
                md +=1
                a = a + len(num_neurons)
                x +=1
                
            if x == 3:
                x=0
                md=0
            
            if c == b + len(num_neurons)*len(learning_rate)+1:
                ml +=1
                b = b + len(num_neurons)*len(learning_rate)
    
    
    """
    Hyperparameter selection for the MLP Model
    """
    
    data = [['CV RMSE [dB] - validation set', '5.05'],
    ['Number of neurons','74'],
    ['Learning rate', '0.1'],
    ['alpha','0.1']]
    
    print(tabulate(data, headers=["Hyperparameters MLP model","Value",]))
    
    
    mlpmodel = MLPRegressor(hidden_layer_sizes=74, 
                                 activation='relu',
                                 solver='lbfgs',
                                 alpha=0.1,
                                 learning_rate_init=0.1,
                                 momentum=0.9,
                                 random_state=1, 
                                 max_iter=6000,
                                 early_stopping=True)
    
    start_time = time.time() #training time in second
    mlpmodel.fit(X_train, np.ravel(y_train))
    time_mlp= time.time() - start_time
    
    #Saving model to know the size in KB of the saved file
    pickle.dump(mlpmodel, open('mlpmodel.pkl','wb'))
    file= r"C:\Users\Yoiz Nuñez\mlpmodel.pkl" #set the local path where the file .pkl is saved
    file_stats=os.stat(file)
    file_KB=file_stats.st_size/(1024)
    
    tot_weigths=74*5 + 74 # weights in the hidden layer + weigths in the output layer
    
    y_pred_train = mlpmodel.predict(X_train)
    y_pred_test = mlpmodel.predict(X_test)
    
    """
    Training set
    """
    #desnormalization to calculate the performance indicators RMSE, MAPE, SD e R2 for the results in Table 4
    y_pred_desn = y_pred_train.reshape(-1,1)
    y_pred_desn_train= scaler.inverse_transform(y_pred_desn)
    y_target_desn_train = scaler.inverse_transform(y_train)
       
    #performance indicators results
    RMSE_train, MAPE_train, SD_train, R2_train= performance_indicators(y_target_desn_train,y_pred_desn_train)
    
    """
    Testing set
    """
    #desnormalization to calculate the performance indicators RMSE, MAPE, SD e R2 for the results in Table 4
    y_pred_desn = y_pred_test.reshape(-1,1)
    y_pred_desn_test= scaler.inverse_transform(y_pred_desn)
    y_target_desn_test = scaler.inverse_transform(y_test)
    
    #performance indicators results
    RMSE_test, MAPE_test, SD_test, R2_test= performance_indicators(y_target_desn_test,y_pred_desn_test)
    
    
    mlp_results={"train_mlp":[y_pred_desn_train,y_target_desn_train,RMSE_train,MAPE_train,SD_train,R2_train],
              "test_mlp":[y_pred_desn_test,y_target_desn_test,RMSE_test,MAPE_test,SD_test,R2_test]}
        
    return mlp_results,tot_weigths,time_mlp,file_KB