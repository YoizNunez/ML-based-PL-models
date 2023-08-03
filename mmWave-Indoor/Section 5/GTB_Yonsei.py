# -*- coding: utf-8 -*-

#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import math

from sklearn.ensemble import GradientBoostingRegressor

import warnings
warnings.filterwarnings("ignore")



def gtb_yonsei(df):    
    """
    Results for Table 12 GTB and empirical PL models performance using the database from Yonsei
    University, Korea.
    """
    
    #----------------------------------------------------------
    #----- Tx Floor 2 for training, and Floor 3 for testing ---
    
    #training floor
    df_1= df.loc[df['Tx Floor']==2]
    df_train= df_1.loc[df['Rx Floor']==2]
    y = df_train.iloc[:, -1]
    
    #testing floor
    df_1= df.loc[df['Tx Floor']==3]
    df_test= df_1.loc[df['Rx Floor']==3]
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
    
    Gradregressor = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=3, min_samples_leaf=4,alpha=0.9, random_state=42,loss='ls', max_features='auto',ccp_alpha=0.00001)
    Gradregressor.fit(X_train,np.ravel(y_train))
    
    y_pred_gb = Gradregressor.predict(X_train)
    y_pred_desn_gb = y_pred_gb*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    y_target_desn_gb = y_train*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_train_f2_f3 = math.sqrt(MSE) 
    
    y_pred_gb = Gradregressor.predict(X_test)
    
    y_pred_desn_gb = y_pred_gb*np.std(y) + np.mean(y)
    y_target_desn_gb = y_test*np.std(y) + np.mean(y)
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_test_f2_f3 = math.sqrt(MSE)
    
    
    #----------------------------------------------------------
    #----- Tx Floor 2 for training, and Floor 4 for testing ---
    
    #training floor
    df_1= df.loc[df['Tx Floor']==2]
    df_train= df_1.loc[df['Rx Floor']==2]
    y = df_train.iloc[:, -1]
    
    #testing floor
    df_1= df.loc[df['Tx Floor']==4]
    df_test= df_1.loc[df['Rx Floor']==4]
    y_test = df_test.iloc[:, -1]
    
    
    # Training
    f = (df_train["f"] - df_train["f"])
    d = (df_train["d"] - np.mean(df_train["d"]))/np.std(df_train["d"])
    nw = (df_train["nw"] - np.mean(df_train["nw"]))/np.std(df_train["nw"])
    
    X_train = pd.concat([d,f,nw], axis=1, join='inner')
    y_train = (df_train["PL"] - np.mean(df_train["PL"]))/np.std(df_train["PL"])
    
    #Testing 
    f = ((df_test["f"] - df_test["f"]))
    d = ((df_test["d"] - min(df_train["d"])) / (max(df_train["d"]) - min(df_train["d"])))
    nw = ((df_test["nw"] - min(df_train["nw"])) / (max(df_train["nw"]) - min(df_train["nw"])))
    
    X_test = pd.concat([d,f,nw], axis=1, join='inner')
    y_test = ((df_test["PL"] - min(df_train["PL"])) /(max(df_train["PL"])- min(df_train["PL"]))) 
    
    
    Gradregressor = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=4, min_samples_leaf=3,alpha=0.9, random_state=42,loss='ls', max_features='auto',ccp_alpha=0.0021)
    Gradregressor.fit(X_train,np.ravel(y_train))
    
    y_pred_gb = Gradregressor.predict(X_train)
    y_pred_desn_gb = y_pred_gb*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    y_target_desn_gb = y_train*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_train_f2_f4 = math.sqrt(MSE) 
    
    y_pred_gb = Gradregressor.predict(X_test)
    
    y_pred_desn_gb = y_pred_gb*np.std(y) + np.mean(y)
    y_target_desn_gb = y_test*np.std(y) + np.mean(y)
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_test_f2_f4 = math.sqrt(MSE)
    
    
    #----------------------------------------------------------
    #----- Tx Floor 3 for training, and Floor 2 for testing ---
    
    #training floor
    df_1= df.loc[df['Tx Floor']==3]
    df_train= df_1.loc[df['Rx Floor']==3]
    y = df_train.iloc[:, -1]
    
    #testing floor
    df_1= df.loc[df['Tx Floor']==2]
    df_test= df_1.loc[df['Rx Floor']==2]
    y_test = df_test.iloc[:, -1]
    
    
    # Training
    f = (df_train["f"] - df_train["f"])
    d = (df_train["d"] - np.mean(df_train["d"]))/np.std(df_train["d"])
    nw = (df_train["nw"] - np.mean(df_train["nw"]))/np.std(df_train["nw"])
    
    X_train = pd.concat([d,f,nw], axis=1, join='inner')
    y_train = (df_train["PL"] - np.mean(df_train["PL"]))/np.std(df_train["PL"])
    
    #Testing 
    f = ((df_test["f"] - df_test["f"]))
    d = ((df_test["d"] - min(df_train["d"])) / (max(df_train["d"]) - min(df_train["d"])))
    nw = ((df_test["nw"] - min(df_train["nw"])) / (max(df_train["nw"]) - min(df_train["nw"])))
    
    X_test = pd.concat([d,f,nw], axis=1, join='inner')
    y_test = ((df_test["PL"] - min(df_train["PL"])) /(max(df_train["PL"])- min(df_train["PL"]))) 
    
    Gradregressor = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, min_samples_leaf=2,alpha=0.9, random_state=42,loss='ls', max_features='auto',ccp_alpha=0.0025)
    Gradregressor.fit(X_train,np.ravel(y_train))
    
    y_pred_gb = Gradregressor.predict(X_train)
    y_pred_desn_gb = y_pred_gb*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    y_target_desn_gb = y_train*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_train_f3_f2 = math.sqrt(MSE) 
    
    y_pred_gb = Gradregressor.predict(X_test)
    
    y_pred_desn_gb = y_pred_gb*np.std(y) + np.mean(y)
    y_target_desn_gb = y_test*np.std(y) + np.mean(y)
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_test_f3_f2 = math.sqrt(MSE)
    
    
    #----------------------------------------------------------
    #----- Tx Floor 3 for training, and Floor 4 for testing ---
    
    #training floor
    df_1= df.loc[df['Tx Floor']==3]
    df_train= df_1.loc[df['Rx Floor']==3]
    y = df_train.iloc[:, -1]
    
    #testing floor
    df_1= df.loc[df['Tx Floor']==4]
    df_test= df_1.loc[df['Rx Floor']==4]
    y_test = df_test.iloc[:, -1]
    
    # Training
    f = (df_train["f"] - df_train["f"])
    d = (df_train["d"] - np.mean(df_train["d"]))/np.std(df_train["d"])
    nw = (df_train["nw"] - np.mean(df_train["nw"]))/np.std(df_train["nw"])
    
    X_train = pd.concat([d,f,nw], axis=1, join='inner')
    y_train = (df_train["PL"] - np.mean(df_train["PL"]))/np.std(df_train["PL"])
    
    #Testing #put the values of the Tx excluded
    f = ((df_test["f"] - df_test["f"]))
    d = ((df_test["d"] - min(df_train["d"])) / (max(df_train["d"]) - min(df_train["d"])))
    nw = ((df_test["nw"] - min(df_train["nw"])) / (max(df_train["nw"]) - min(df_train["nw"])))
    
    X_test = pd.concat([d,f,nw], axis=1, join='inner')
    y_test = ((df_test["PL"] - min(df_train["PL"])) /(max(df_train["PL"])- min(df_train["PL"]))) 
    
    Gradregressor = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, min_samples_leaf=2,alpha=0.9, random_state=42,loss='ls', max_features='auto',ccp_alpha=0.0025)
    Gradregressor.fit(X_train,np.ravel(y_train))
    
    y_pred_gb = Gradregressor.predict(X_train)
    y_pred_desn_gb = y_pred_gb*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    y_target_desn_gb = y_train*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_train_f3_f4 = math.sqrt(MSE) 
    
    y_pred_gb = Gradregressor.predict(X_test)
    
    y_pred_desn_gb = y_pred_gb*np.std(y) + np.mean(y)
    y_target_desn_gb = y_test*np.std(y) + np.mean(y)
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_test_f3_f4 = math.sqrt(MSE)
    
    #----------------------------------------------------------
    #----- Tx Floor 4 for training, and Floor 2 for testing ---
    
    #training floor
    df_1= df.loc[df['Tx Floor']==4]
    df_train= df_1.loc[df['Rx Floor']==4]
    y = df_train.iloc[:, -1]
    
    #testing floor
    df_1= df.loc[df['Tx Floor']==2]
    df_test= df_1.loc[df['Rx Floor']==2]
    y_test = df_test.iloc[:, -1]
    
    # Training
    f = (df_train["f"] - df_train["f"])
    d = (df_train["d"] - np.mean(df_train["d"]))/np.std(df_train["d"])
    nw = (df_train["nw"] - np.mean(df_train["nw"]))/np.std(df_train["nw"])
    
    X_train = pd.concat([d,f,nw], axis=1, join='inner')
    y_train = (df_train["PL"] - np.mean(df_train["PL"]))/np.std(df_train["PL"])
    
    #Testing 
    f = ((df_test["f"] - df_test["f"]))
    d = ((df_test["d"] - min(df_train["d"])) / (max(df_train["d"]) - min(df_train["d"])))
    nw = ((df_test["nw"] - min(df_train["nw"])) / (max(df_train["nw"]) - min(df_train["nw"])))
    
    X_test = pd.concat([d,f,nw], axis=1, join='inner')
    y_test = ((df_test["PL"] - min(df_train["PL"])) /(max(df_train["PL"])- min(df_train["PL"]))) 
    
    Gradregressor = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, min_samples_leaf=2,alpha=0.9, random_state=42,loss='ls', max_features='auto',ccp_alpha=0.002)
    Gradregressor.fit(X_train,np.ravel(y_train))
    
    y_pred_gb = Gradregressor.predict(X_train)
    y_pred_desn_gb = y_pred_gb*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    y_target_desn_gb = y_train*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_train_f4_f2 = math.sqrt(MSE) 
    
    y_pred_gb = Gradregressor.predict(X_test)
    
    y_pred_desn_gb = y_pred_gb*np.std(y) + np.mean(y)
    y_target_desn_gb = y_test*np.std(y) + np.mean(y)
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_test_f4_f2 = math.sqrt(MSE)
    
    #----------------------------------------------------
    #2) Tx Floor 4 for training, and Floor 3 for testing
    #training floor
    df_1= df.loc[df['Tx Floor']==4]
    df_train= df_1.loc[df['Rx Floor']==4]
    y = df_train.iloc[:, -1]
    
    #testing floor
    df_1= df.loc[df['Tx Floor']==3]
    df_test= df_1.loc[df['Rx Floor']==3]
    y_test = df_test.iloc[:, -1]
    
    # Training
    f = (df_train["f"] - df_train["f"])
    d = (df_train["d"] - np.mean(df_train["d"]))/np.std(df_train["d"])
    nw = (df_train["nw"] - np.mean(df_train["nw"]))/np.std(df_train["nw"])
    
    X_train = pd.concat([d,f,nw], axis=1, join='inner')
    y_train = (df_train["PL"] - np.mean(df_train["PL"]))/np.std(df_train["PL"])
    
    #Testing #put the values of the Tx excluded
    f = ((df_test["f"] - df_test["f"]))
    d = ((df_test["d"] - min(df_train["d"])) / (max(df_train["d"]) - min(df_train["d"])))
    nw = ((df_test["nw"] - min(df_train["nw"])) / (max(df_train["nw"]) - min(df_train["nw"])))
    
    X_test = pd.concat([d,f,nw], axis=1, join='inner')
    y_test = ((df_test["PL"] - min(df_train["PL"])) /(max(df_train["PL"])- min(df_train["PL"]))) 
    
    Gradregressor = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, min_samples_leaf=2,alpha=0.9, random_state=42,loss='ls', max_features='auto',ccp_alpha=0.002)
    Gradregressor.fit(X_train,np.ravel(y_train))
    
    y_pred_gb = Gradregressor.predict(X_train)
    y_pred_desn_gb = y_pred_gb*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    y_target_desn_gb = y_train*np.std(df_train["PL"]) + np.mean(df_train["PL"])
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_train_f4_f3 = math.sqrt(MSE) 
    
    y_pred_gb = Gradregressor.predict(X_test)
    
    y_pred_desn_gb = y_pred_gb*np.std(y) + np.mean(y)
    y_target_desn_gb = y_test*np.std(y) + np.mean(y)
    
    MSE = np.square(np.subtract(y_target_desn_gb,y_pred_desn_gb)).mean()
    RMSE_test_f4_f3 = math.sqrt(MSE)
    
    results_yonsei = {"train_floor2 - test_floor3_4":[RMSE_train_f2_f3,RMSE_test_f2_f3,RMSE_test_f2_f4],
                      "train_floor3 - test_floor2_4":[RMSE_train_f3_f2,RMSE_test_f3_f2,RMSE_test_f3_f4], 
                      "train_floor4 - test_floor2_3":[RMSE_train_f4_f2,RMSE_test_f4_f2,RMSE_test_f4_f3],
        
        }
    

    return results_yonsei
