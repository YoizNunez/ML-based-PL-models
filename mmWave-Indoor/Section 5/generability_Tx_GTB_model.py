# -*- coding: utf-8 -*-

#imprting libraries
import numpy as np
import pandas as pd
import seaborn as sns

# libraries for data preprocessing 
from sklearn.preprocessing import MinMaxScaler    
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score
import math

from sklearn.ensemble import GradientBoostingRegressor

import tabulate
from tabulate import tabulate

"""
Code for Table 11: Generalization capability analysis for the GTB-based PL model. The models
are trained using two transmitters and evaluated (tested) on the four remaining ones.
"""

def generalization_gtb_model_tx(df):
    
    #---------Experiment 1-----------
    #training Tx
    df_train= df.loc[(df['Tx']=='Tx1') | (df['Tx']=='Tx3')] #In this case, it will be used the Tx1 and Tx3 for training 
    y = df_train.iloc[:, -1]
    
    #testing Tx
    df_test= df.loc[(df['Tx']=='Tx2') | (df['Tx']=='Tx4') | (df['Tx']=='Tx5') | (df['Tx']=='Tx6')] #and, it will be used the Tx2, Tx4,Tx5 and Tx6 for testing 
    y_test = df_test.iloc[:, -1]
    
    f = (df_train["f"] - df_train["f"])
    d = (df_train["d"] - np.mean(df_train["d"]))/np.std(df_train["d"])
    nw = (df_train["nw"] - np.mean(df_train["nw"]))/np.std(df_train["nw"])
    
    X_train = pd.concat([d,f,nw], axis=1, join='inner')
    y_train = (df_train["PL"] - np.mean(df_train["PL"]))/np.std(df_train["PL"])
    
    f = ((df_test["f"] - df_test["f"]))
    d = ((df_test["d"] - min(df_train["d"])) / (max(df_train["d"]) - min(df_train["d"])))
    nw = ((df_test["nw"] - min(df_train["nw"])) / (max(df_train["nw"]) - min(df_train["nw"])))
    
    X_test = pd.concat([d,f,nw], axis=1, join='inner')
    y_test = ((df_test["PL"] - min(df_train["PL"])) /(max(df_train["PL"])- min(df_train["PL"]))) 
    
    f = (df_train["f"] - np.mean(df_train["f"]))/np.std(df_train["f"])
    d = (df_train["d"] - np.mean(df_train["d"]))/np.std(df_train["d"])
    nw = (df_train["nw"] - np.mean(df_train["nw"]))/np.std(df_train["nw"])
    
    X_train = pd.concat([d,f,nw], axis=1, join='inner')
    y_train = (df_train["PL"] - np.mean(df_train["PL"]))/np.std(df_train["PL"])
    
    f = ((df_test["f"] - df_test["f"]))
    d = ((df_test["d"] - min(df_train["d"])) / (max(df_train["d"]) - min(df_train["d"])))
    nw = ((df_test["nw"] - min(df_train["nw"])) / (max(df_train["nw"]) - min(df_train["nw"])))
        
    X_test = pd.concat([d,f,nw], axis=1, join='inner')
    y_test = ((df_test["PL"] - min(df_train["PL"])) /(max(df_train["PL"])- min(df_train["PL"]))) 
    
    Gradregressor = GradientBoostingRegressor(n_estimators=246, learning_rate=0.1, max_depth=2, min_samples_leaf=3,alpha=0.9, random_state=42,loss='ls', max_features='auto',ccp_alpha=0.00001)
    Gradregressor.fit(X_train,np.ravel(y_train))
    
    y_pred_gb = Gradregressor.predict(X_train)
    y_pred_desn_gb = y_pred_gb*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    y_target_desn_gb = y_train*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_train_tx1 = math.sqrt(MSE) 
    
    y_pred_gb = Gradregressor.predict(X_test)
    
    y_pred_desn_gb = y_pred_gb*np.std(y) + np.mean(y)
    y_target_desn_gb = y_test*np.std(y) + np.mean(y)
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_test_tx1 = math.sqrt(MSE)
    
    
    #---------Experiment 2-----------
    #training Tx
    df_train= df.loc[(df['Tx']=='Tx1') | (df['Tx']=='Tx4')] # exclude the Tx in the training set to be assessed in the testing set
    y = df_train.iloc[:, -1]
    
    #testing Tx
    df_test= df.loc[(df['Tx']=='Tx2') | (df['Tx']=='Tx3') | (df['Tx']=='Tx5') | (df['Tx']=='Tx6')]
    y_test = df_test.iloc[:, -1]
    
    f = (df_train["f"] - df_train["f"])
    d = (df_train["d"] - np.mean(df_train["d"]))/np.std(df_train["d"])
    nw = (df_train["nw"] - np.mean(df_train["nw"]))/np.std(df_train["nw"])
    
    X_train = pd.concat([d,f,nw], axis=1, join='inner')
    y_train = (df_train["PL"] - np.mean(df_train["PL"]))/np.std(df_train["PL"])
    
    f = ((df_test["f"] - df_test["f"]))
    d = ((df_test["d"] - min(df_train["d"])) / (max(df_train["d"]) - min(df_train["d"])))
    nw = ((df_test["nw"] - min(df_train["nw"])) / (max(df_train["nw"]) - min(df_train["nw"])))
    
    X_test = pd.concat([d,f,nw], axis=1, join='inner')
    y_test = ((df_test["PL"] - min(df_train["PL"])) /(max(df_train["PL"])- min(df_train["PL"]))) 
    
    f = (df_train["f"] - np.mean(df_train["f"]))/np.std(df_train["f"])
    d = (df_train["d"] - np.mean(df_train["d"]))/np.std(df_train["d"])
    nw = (df_train["nw"] - np.mean(df_train["nw"]))/np.std(df_train["nw"])
    
    X_train = pd.concat([d,f,nw], axis=1, join='inner')
    y_train = (df_train["PL"] - np.mean(df_train["PL"]))/np.std(df_train["PL"])
    
    f = ((df_test["f"] - df_test["f"]))
    d = ((df_test["d"] - min(df_train["d"])) / (max(df_train["d"]) - min(df_train["d"])))
    nw = ((df_test["nw"] - min(df_train["nw"])) / (max(df_train["nw"]) - min(df_train["nw"])))
        
    X_test = pd.concat([d,f,nw], axis=1, join='inner')
    y_test = ((df_test["PL"] - min(df_train["PL"])) /(max(df_train["PL"])- min(df_train["PL"]))) 
    
    Gradregressor = GradientBoostingRegressor(n_estimators=140, learning_rate=0.1, max_depth=3, min_samples_leaf=10,alpha=0.9, random_state=42,loss='ls', max_features='auto',ccp_alpha=0.003)
    Gradregressor.fit(X_train,np.ravel(y_train))
    
    y_pred_gb = Gradregressor.predict(X_train)
    y_pred_desn_gb = y_pred_gb*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    y_target_desn_gb = y_train*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_train_tx2 = math.sqrt(MSE) 
    
    y_pred_gb = Gradregressor.predict(X_test)
    
    y_pred_desn_gb = y_pred_gb*np.std(y) + np.mean(y)
    y_target_desn_gb = y_test*np.std(y) + np.mean(y)
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_test_tx2 = math.sqrt(MSE)
    
    #---------Experiment 3-----------
    #training Tx
    df_train= df.loc[(df['Tx']=='Tx2') | (df['Tx']=='Tx4')] # exclude the Tx in the training set to be assessed in the testing set
    y = df_train.iloc[:, -1]
    
    #testing Tx
    df_test= df.loc[(df['Tx']=='Tx1') | (df['Tx']=='Tx3') | (df['Tx']=='Tx5') | (df['Tx']=='Tx6')]
    y_test = df_test.iloc[:, -1]
    
    f = (df_train["f"] - df_train["f"])
    d = (df_train["d"] - np.mean(df_train["d"]))/np.std(df_train["d"])
    nw = (df_train["nw"] - np.mean(df_train["nw"]))/np.std(df_train["nw"])
    
    X_train = pd.concat([d,f,nw], axis=1, join='inner')
    y_train = (df_train["PL"] - np.mean(df_train["PL"]))/np.std(df_train["PL"])
    
    f = ((df_test["f"] - df_test["f"]))
    d = ((df_test["d"] - min(df_train["d"])) / (max(df_train["d"]) - min(df_train["d"])))
    nw = ((df_test["nw"] - min(df_train["nw"])) / (max(df_train["nw"]) - min(df_train["nw"])))
    
    X_test = pd.concat([d,f,nw], axis=1, join='inner')
    y_test = ((df_test["PL"] - min(df_train["PL"])) /(max(df_train["PL"])- min(df_train["PL"]))) 
    
    f = (df_train["f"] - np.mean(df_train["f"]))/np.std(df_train["f"])
    d = (df_train["d"] - np.mean(df_train["d"]))/np.std(df_train["d"])
    nw = (df_train["nw"] - np.mean(df_train["nw"]))/np.std(df_train["nw"])
    
    X_train = pd.concat([d,f,nw], axis=1, join='inner')
    y_train = (df_train["PL"] - np.mean(df_train["PL"]))/np.std(df_train["PL"])
    
    f = ((df_test["f"] - df_test["f"]))
    d = ((df_test["d"] - min(df_train["d"])) / (max(df_train["d"]) - min(df_train["d"])))
    nw = ((df_test["nw"] - min(df_train["nw"])) / (max(df_train["nw"]) - min(df_train["nw"])))
        
    X_test = pd.concat([d,f,nw], axis=1, join='inner')
    y_test = ((df_test["PL"] - min(df_train["PL"])) /(max(df_train["PL"])- min(df_train["PL"]))) 
    
    Gradregressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=2, min_samples_leaf=10,alpha=0.9, random_state=42,loss='ls', max_features='auto',ccp_alpha=0.001)
    Gradregressor.fit(X_train,np.ravel(y_train))
    
    y_pred_gb = Gradregressor.predict(X_train)
    y_pred_desn_gb = y_pred_gb*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    y_target_desn_gb = y_train*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_train_tx3 = math.sqrt(MSE) 
    
    y_pred_gb = Gradregressor.predict(X_test)
    
    y_pred_desn_gb = y_pred_gb*np.std(y) + np.mean(y)
    y_target_desn_gb = y_test*np.std(y) + np.mean(y)
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_test_tx3 = math.sqrt(MSE)
    
    
    #---------Experiment 4-----------
    #training Tx
    df_train= df.loc[(df['Tx']=='Tx2') | (df['Tx']=='Tx6')] # exclude the Tx in the training set to be assessed in the testing set
    y = df_train.iloc[:, -1]
    
    #testing Tx
    df_test= df.loc[(df['Tx']=='Tx1') | (df['Tx']=='Tx3') | (df['Tx']=='Tx4') | (df['Tx']=='Tx5')]
    y_test = df_test.iloc[:, -1]
    
    f = (df_train["f"] - df_train["f"])
    d = (df_train["d"] - np.mean(df_train["d"]))/np.std(df_train["d"])
    nw = (df_train["nw"] - np.mean(df_train["nw"]))/np.std(df_train["nw"])
    
    X_train = pd.concat([d,f,nw], axis=1, join='inner')
    y_train = (df_train["PL"] - np.mean(df_train["PL"]))/np.std(df_train["PL"])
    
    f = ((df_test["f"] - df_test["f"]))
    d = ((df_test["d"] - min(df_train["d"])) / (max(df_train["d"]) - min(df_train["d"])))
    nw = ((df_test["nw"] - min(df_train["nw"])) / (max(df_train["nw"]) - min(df_train["nw"])))
    
    X_test = pd.concat([d,f,nw], axis=1, join='inner')
    y_test = ((df_test["PL"] - min(df_train["PL"])) /(max(df_train["PL"])- min(df_train["PL"]))) 
    
    f = (df_train["f"] - np.mean(df_train["f"]))/np.std(df_train["f"])
    d = (df_train["d"] - np.mean(df_train["d"]))/np.std(df_train["d"])
    nw = (df_train["nw"] - np.mean(df_train["nw"]))/np.std(df_train["nw"])
    
    X_train = pd.concat([d,f,nw], axis=1, join='inner')
    y_train = (df_train["PL"] - np.mean(df_train["PL"]))/np.std(df_train["PL"])
    
    f = ((df_test["f"] - df_test["f"]))
    d = ((df_test["d"] - min(df_train["d"])) / (max(df_train["d"]) - min(df_train["d"])))
    nw = ((df_test["nw"] - min(df_train["nw"])) / (max(df_train["nw"]) - min(df_train["nw"])))
        
    X_test = pd.concat([d,f,nw], axis=1, join='inner')
    y_test = ((df_test["PL"] - min(df_train["PL"])) /(max(df_train["PL"])- min(df_train["PL"]))) 
    
    Gradregressor = GradientBoostingRegressor(n_estimators=30, learning_rate=0.1, max_depth=3, min_samples_leaf=10,alpha=0.9, random_state=42,loss='ls', max_features='auto',ccp_alpha=0.001)
    Gradregressor.fit(X_train,np.ravel(y_train))
    
    y_pred_gb = Gradregressor.predict(X_train)
    y_pred_desn_gb = y_pred_gb*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    y_target_desn_gb = y_train*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_train_tx4 = math.sqrt(MSE) 
    
    y_pred_gb = Gradregressor.predict(X_test)
    
    y_pred_desn_gb = y_pred_gb*np.std(y) + np.mean(y)
    y_target_desn_gb = y_test*np.std(y) + np.mean(y)
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_test_tx4 = math.sqrt(MSE)
    
    results_gtb_gen_tx = {"Tx 1":[RMSE_train_tx1,RMSE_test_tx1],
                       "Tx 2":[RMSE_train_tx2,RMSE_test_tx2], 
                       "Tx 3":[RMSE_train_tx3,RMSE_test_tx3],
                       "Tx 4":[RMSE_train_tx4,RMSE_test_tx4]        
        }

    return results_gtb_gen_tx