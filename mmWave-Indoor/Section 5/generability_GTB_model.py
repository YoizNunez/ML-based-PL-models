# -*- coding: utf-8 -*-

#importing libraries
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


def generalization_gtb_model(df):
        
    #------ Tx 1 is excluded from the training set, and it is used in the testing set------
    #training Tx
    df_train= df.loc[df['Tx']!='Tx1'] # exclude the Tx in the training set to be assessed in the testing set
    y = df_train.iloc[:, -1]
    
    #testing Tx
    df_test= df.loc[df['Tx']=='Tx1']
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
    
    Gradregressor = GradientBoostingRegressor(n_estimators=161, learning_rate=0.1, max_depth=6, min_samples_leaf=2,alpha=0.9, random_state=42,loss='ls', max_features='auto',ccp_alpha=0.001)
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
    
    
    #------ Tx 2 is excluded from the training set, and it is used in the testing set -------
    #training Tx
    df_train= df.loc[df['Tx']!='Tx2'] # exclude the Tx in the training set to be assessed in the testing set
    y = df_train.iloc[:, -1]
    
    #testing Tx
    df_test= df.loc[df['Tx']=='Tx2']
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
    
    Gradregressor = GradientBoostingRegressor(n_estimators=246, learning_rate=0.1, max_depth=4, min_samples_leaf=2,alpha=0.9, random_state=42,loss='ls', max_features='auto',ccp_alpha=0.0003)
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
    
    
    #------ Tx 3 is excluded from the training set, and it is used in the testing set -------
    #training Tx
    df_train= df.loc[df['Tx']!='Tx3'] # exclude the Tx in the training set to be assessed in the testing set
    y = df_train.iloc[:, -1]
    
    #testing Tx
    df_test= df.loc[df['Tx']=='Tx3']
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
    
    Gradregressor = GradientBoostingRegressor(n_estimators=246, learning_rate=0.07, max_depth=2, min_samples_leaf=2,alpha=0.9, random_state=42,loss='ls', max_features='auto',ccp_alpha=0.00025)
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
    
    
    #------ Tx 4 is excluded from the training set, and it is used in the testing set ------
    #training Tx
    df_train= df.loc[df['Tx']!='Tx4'] # exclude the Tx in the training set to be assessed in the testing set
    y = df_train.iloc[:, -1]
    
    #testing Tx
    df_test= df.loc[df['Tx']=='Tx4']
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
    
    Gradregressor = GradientBoostingRegressor(n_estimators=246, learning_rate=0.10, max_depth=3, min_samples_leaf=2,alpha=0.9, random_state=42,loss='ls', max_features='auto',ccp_alpha=0.0002)
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
    
    
    #------ Tx 5 is excluded from the training set, and it is used in the testing set ------
    #training Tx
    df_train= df.loc[df['Tx']!='Tx5'] # exclude the Tx in the training set to be assessed in the testing set
    y = df_train.iloc[:, -1]
    
    #testing Tx
    df_test= df.loc[df['Tx']=='Tx5']
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
    
    Gradregressor = GradientBoostingRegressor(n_estimators=246, learning_rate=0.05, max_depth=3, min_samples_leaf=2,alpha=0.9, random_state=42,loss='ls', max_features='auto',ccp_alpha=0.0007)
    Gradregressor.fit(X_train,np.ravel(y_train))
    
    y_pred_gb = Gradregressor.predict(X_train)
    y_pred_desn_gb = y_pred_gb*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    y_target_desn_gb = y_train*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_train_tx5 = math.sqrt(MSE)
    
    y_pred_gb = Gradregressor.predict(X_test)
    
    y_pred_desn_gb = y_pred_gb*np.std(y) + np.mean(y)
    y_target_desn_gb = y_test*np.std(y) + np.mean(y)
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_test_tx5 = math.sqrt(MSE)
    
    
    #------ Tx 6 is excluded from the training set, and it is used in the testing set -------
    #training Tx
    df_train= df.loc[df['Tx']!='Tx6'] # exclude the Tx in the training set to be assessed in the testing set
    y = df_train.iloc[:, -1]
    
    #testing Tx
    df_test= df.loc[df['Tx']=='Tx6']
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
    
    Gradregressor = GradientBoostingRegressor(n_estimators=246, learning_rate=0.05, max_depth=3, min_samples_leaf=3,alpha=0.9, random_state=42,loss='ls', max_features='auto',ccp_alpha=0.0002)
    Gradregressor.fit(X_train,np.ravel(y_train))
    
    y_pred_gb = Gradregressor.predict(X_train)
    y_pred_desn_gb = y_pred_gb*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    y_target_desn_gb = y_train*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_train_tx6 = math.sqrt(MSE)
    
    y_pred_gb = Gradregressor.predict(X_test)
    
    y_pred_desn_gb = y_pred_gb*np.std(y) + np.mean(y)
    y_target_desn_gb = y_test*np.std(y) + np.mean(y)
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_test_tx6 = math.sqrt(MSE)
    
    results_gtb_gen = {"Tx 1":[RMSE_train_tx1,RMSE_test_tx1],
                       "Tx 2":[RMSE_train_tx2,RMSE_test_tx2], 
                       "Tx 3":[RMSE_train_tx3,RMSE_test_tx3],
                       "Tx 4":[RMSE_train_tx4,RMSE_test_tx4],
                       "Tx 5":[RMSE_train_tx5,RMSE_test_tx5],
                       "Tx 6":[RMSE_train_tx6,RMSE_test_tx6]
        }
     
    return results_gtb_gen
