# -*- coding: utf-8 -*-

"""
Code for the performance, mutual infomation and interpretability machine learning techiques 
for Table 6 for each predictors coalitions for the SVR model
"""

#importing libraries
from utils_tools_sec5 import functions_tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# libraries for data preprocessing  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.svm import SVR
import knncmi as knn
import pickle
import itertools
import math

import pymint #version: 0.2.6

import random
random.seed(0)

import warnings
warnings.filterwarnings("ignore")


def svr_model_IML(df):
    
    print("----Calculating performance indicators SVR model---")
    """
    training the ML model
    """
    def svr(X_train,y_train):
      svr_regressor = SVR(kernel='rbf', C=2100, epsilon=0.1,gamma=0.2)
      svr_regressor.fit(X_train,np.ravel(y_train))
      return svr_regressor
    
    
    """
    function to compute the output prediction, with desnormalized values
    """
    def path_loss_svr(X_train,y_train,X_test,y_test):
    
      svr_regressor = svr(X_train,y_train)
      svr_regressor.fit(X_train,np.ravel(y_train))
    
      # predicted path-loss value for the training and testing set
      y_pred_train= svr_regressor.predict(X_train)
      y_pred_test= svr_regressor.predict(X_test)
    
      # desnormalization
      y_pred_train = y_pred_train.reshape(-1,1)
      y_pred_train = scaler.inverse_transform(y_pred_train) 
      y_target_train = scaler.inverse_transform(y_train) 
    
      y_pred_test = y_pred_test.reshape(-1,1)
      y_pred_test = scaler.inverse_transform(y_pred_test) 
      y_target_test = scaler.inverse_transform(y_test) 
    
      return y_target_train,y_target_test,y_pred_train,y_pred_test
    
    """
    To initialize the function scaler
    """
    X = df.iloc[:, [0,1,2,3,4]]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # normalization of numerical input
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    # normalize output
    y_train = y_train.reshape(-1, 1)
    y_train = scaler.fit_transform(y_train)
    y_test = y_test.reshape(-1, 1)
    y_test = scaler.transform(y_test)
    
    # convert Output Variable to Float
    y_train, y_test = y_train.astype(float), y_test.astype(float),
    
    p=[0,1,2,3,4] #list of predictor indices
    
    #matrix to save the performance indicators 
    rmse_1=np.zeros((5, 6))
    rmse_2=np.zeros((5, 6))
    rmse_3=np.zeros((5, 6))
    rmse_4=np.zeros((5, 6))
    rmse_5=np.zeros((5, 6))
    
    """
    Choosing predictor 1. That is, the predictor with the lowest RMSE value among the five predictors.
    """
    for i in range(len(p)):  
    
      y = df.iloc[:,-1] #output PL
    
      X_train, X_test, y_train, y_test = functions_tools.correct_normalize_one(p[i],df,y)
      
      y_target_train,y_target_test,y_pred_train,y_pred_test = path_loss_svr(X_train,y_train,X_test,y_test)
      RMSE_train, MAPE_train, SD_train, R2_train = functions_tools.statistics(y_target_train, y_pred_train)
      RMSE_test, MAPE, SD, R2 = functions_tools.statistics(y_target_test, y_pred_test)
    
      rmse_1[i][0] = i
      rmse_1[i][1] = RMSE_train
      rmse_1[i][2] = RMSE_test
      rmse_1[i][3] = MAPE_train
      rmse_1[i][4] = SD_train
      rmse_1[i][5] = R2_train
      
    #choose the minimum rmse value in the matrix rmse_1
    p_idx = rmse_1[:,1].argmin() #min value in column 1
    pred1 = rmse_1[p_idx][0].astype(np.int64) 
    rmse_p1 = rmse_1[p_idx][1]
    mape_p1 = rmse_1[p_idx][3]
    SD_p1 = rmse_1[p_idx][4]
    R2_p1 = rmse_1[p_idx][5]
    
    """
    Choosing predictor 2
    """
    for i in range(len(p)):  
        
      X_train, X_test, y_train, y_test = functions_tools.correct_normalize_two(pred1,p[i],df,y)
    
      y_target_train,y_target_test,y_pred_train,y_pred_test = path_loss_svr(X_train,y_train,X_test,y_test)
      RMSE_train, MAPE_train, SD_train, R2_train = functions_tools.statistics(y_target_train, y_pred_train)
      RMSE_test, MAPE, SD, R2 = functions_tools.statistics(y_target_test, y_pred_test)
    
      rmse_2[i][0] = i
      rmse_2[i][1] = RMSE_train
      rmse_2[i][2] = RMSE_test
      rmse_2[i][3] = MAPE_train
      rmse_2[i][4] = SD_train
      rmse_2[i][5] = R2_train
    
    #choose the min rmse value
    p_idx = rmse_2[:,1].argmin() #min value in column 1
    pred2 = rmse_2[p_idx][0].astype(np.int64) 
    rmse_p2 = rmse_2[p_idx][1]
    mape_p2 = rmse_2[p_idx][3]
    SD_p2 = rmse_2[p_idx][4]
    R2_p2= rmse_2[p_idx][5]
    
    
    """
    Choosing predictor 3
    """
    for i in range(len(p)):  
     
      X_train, X_test, y_train, y_test= functions_tools.correct_normalize_three(pred1,pred2,p[i],df,y)
      
      y_target_train,y_target_test,y_pred_train,y_pred_test = path_loss_svr(X_train,y_train,X_test,y_test)
      RMSE_train, MAPE_train, SD_train, R2_train = functions_tools.statistics(y_target_train, y_pred_train)
      RMSE_test, MAPE, SD, R2 = functions_tools.statistics(y_target_test, y_pred_test)
    
      rmse_3[i][0] = i
      rmse_3[i][1] = RMSE_train
      rmse_3[i][2] = RMSE_test
      rmse_3[i][3] = MAPE_train
      rmse_3[i][4] = SD_train
      rmse_3[i][5] = R2_train
    
    #choose the min rmse value
    p_idx = rmse_3[:,1].argmin() #min value in column 1
    pred3 = rmse_3[p_idx][0].astype(np.int64) 
    rmse_p3 = rmse_3[p_idx][1]
    mape_p3 = rmse_3[p_idx][3]
    SD_p3 = rmse_3[p_idx][4]
    R2_p3= rmse_3[p_idx][5]
    
    """
    Choosing predictor 4
    """
    for i in range(len(p)):  
    
      X_train, X_test, y_train, y_test = functions_tools.correct_normalize_four(pred1,pred2,pred3,p[i],df,y)
      
      y_target_train,y_target_test,y_pred_train,y_pred_test = path_loss_svr(X_train,y_train,X_test,y_test)
      RMSE_train, MAPE_train, SD_train, R2_train = functions_tools.statistics(y_target_train, y_pred_train)
      RMSE_test, MAPE, SD, R2 = functions_tools.statistics(y_target_test, y_pred_test)
    
      rmse_4[i][0] = i
      rmse_4[i][1] = RMSE_train
      rmse_4[i][2] = RMSE_test
      rmse_4[i][3] = MAPE_train
      rmse_4[i][4] = SD_train
      rmse_4[i][5] = R2_train
    
    #choose the min rmse value
    p_idx = rmse_4[:,1].argmin() #min value in column 1
    pred4 = rmse_4[p_idx][0].astype(np.int64) 
    rmse_p4 = rmse_4[p_idx][1]
    mape_p4 = rmse_4[p_idx][3]
    SD_p4 = rmse_4[p_idx][4]
    R2_p4= rmse_4[p_idx][5]
    
    
    """
    choosing final predictor
    """
    pred5=1 #the last predictor to be assessed for the ANN model
    
    X_train, X_test, y_train, y_test = functions_tools.correct_normalize_five(pred1,pred2,pred3,pred4,pred5,df,y)
    y_target_train,y_target_test,y_pred_train,y_pred_test = path_loss_svr(X_train,y_train,X_test,y_test)
    RMSE_train, MAPE_train, SD_train, R2_train = functions_tools.statistics(y_target_train, y_pred_train)
    RMSE_test, MAPE_test, SD_test, R2_test = functions_tools.statistics(y_target_test, y_pred_test)
    
    rmse_p5 = RMSE_train
    mape_p5 = MAPE_train
    SD_p5 = SD_train
    R2_p5= R2_train
    
    svr_results={"Pred1":[pred1,rmse_p1,mape_p1,SD_p1,R2_p1],
                 "Pred2":[pred2,rmse_p2,mape_p2,SD_p2,R2_p2],
                 "Pred3":[pred3,rmse_p3,mape_p3,SD_p3,R2_p3],
                 "Pred4":[pred4,rmse_p4,mape_p4,SD_p4,R2_p4],
                 "Pred5":[pred5,rmse_p5,mape_p5,SD_p5,R2_p5]
                 }
        
    """
    Mutual information results shown in Table 6
    """
    print("----Calculating Mutual Information SVR model---")
    X = df.iloc[:, [0,1,2,3,4]]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y = y_train
    x = X_train.iloc[:,[0,1,2,3,4]]
    df1 = pd.concat([x,y], axis=1, join='inner')
    
    #Calculating JMI
    jmi_2= knn.cmi(['nw','dy'],['PL'],[],33, df1)
    jmi_3= knn.cmi(['nw','dy','dx'],['PL'],[],29, df1) 
    jmi_4= knn.cmi(['nw','dy','dx','f'],['PL'],[],31, df1) 
    jmi_5= knn.cmi(['nw','dy','dx','f','d'],['PL'],[],31, df1) 
     
    #Calculating CMI
    #cmi(X;Y|Z), mutual information between X and Y, conditioned in Z. 
    cmi_2= knn.cmi(['PL'],['dy'],['nw'],45, df1) 
    cmi_3= knn.cmi(['PL'],['dx'],['nw','dy'],36, df1) 
    cmi_4= knn.cmi(['PL'],['f'],['nw','dy','dx'],47, df1)
    cmi_5= knn.cmi(['PL'],['d'],['nw','dy','dx','f'],60, df1)
     
    mi_results_svr={"Subset 2 predictors":[jmi_2,cmi_2],
                "Subset 3 predictors":[jmi_3,cmi_3],
                "Subset 4 predictors":[jmi_4,cmi_4],
                "Subset 5 predictors":[jmi_5,cmi_5]
                 }
    
    """
    Interpretability Machine Learning indicators
    """
    print("----Calculating IML indicators SVR model---")

    y = df.iloc[:, -1]
    
    #preparing the input's model for each subset
    pred1 = svr_results["Pred1"][0]
    pred2 = svr_results["Pred2"][0]
    pred3 = svr_results["Pred3"][0]
    pred4 = svr_results["Pred4"][0]
    pred5 = svr_results["Pred5"][0]
    
    X_train_p1, X_test_p1, y_train_p1, y_test_p1 = functions_tools.correct_normalize_one(pred1,df,y)
    X_train_p2, X_test_p2, y_train_p2, y_test_p2 = functions_tools.correct_normalize_two(pred1,pred2,df,y)
    X_train_p3, X_test_p3, y_train_p3, y_test_p3 = functions_tools.correct_normalize_three(pred1,pred2,pred3,df,y)
    X_train_p4, X_test_p4, y_train_p4, y_test_p4 = functions_tools.correct_normalize_four(pred1,pred2,pred3,pred4,df,y)
    
    X_train_p5,X_test_p5, y_train_p5, y_test_p5 = functions_tools.correct_normalize_five(pred1,pred2,pred3,pred4,pred5,df,y)
    
    #name of predictors
    features_list_p1=['nw']
    features_list_p2=['nw','dy']
    features_list_p3=['nw','dy','dx']
    features_list_p4=['nw','dy','dx','f']
    features_list_p5=['nw','dy','dx','f','d']
    
    X_train_df_p1 = pd.DataFrame(X_train_p1, columns = features_list_p1)
    X_train_df_p1 = X_train_df_p1.astype({'nw': 'category'})
    X_train_df_p2 = pd.DataFrame(X_train_p2, columns = features_list_p2)
    X_train_df_p2 = X_train_df_p2.astype({'nw': 'category'})
    X_train_df_p3 = pd.DataFrame(X_train_p3, columns = features_list_p3)
    X_train_df_p3 = X_train_df_p3.astype({'nw': 'category'})
    X_train_df_p4 = pd.DataFrame(X_train_p4, columns = features_list_p4)
    X_train_df_p4 = X_train_df_p4.astype({'nw': 'category'}) #for the categorical input, we define it as category input
    X_train_df_p5 = pd.DataFrame(X_train_p5, columns = features_list_p5)
    X_train_df_p5 = X_train_df_p5.astype({'nw': 'category'})
    
    #function to compute the IML tools
    
    def IML(X_train,y_train,X_train_df,features_list):
    
        svr_regressor = SVR(kernel='rbf',  C=2100, epsilon=0.1,gamma=0.2)
        svr_regressor.fit(X_train, np.ravel(y_train))
        
        # saving the model as pickle file to be reading by the library pymint, according its especification.
        pickle.dump(svr_regressor, open('svrmodel_IML.pkl','wb')) #save the model, i.e., 'mlpmodel_IML_5.pkl' for the subset of 5 predictors
        svr = pickle.load(open('svrmodel_IML.pkl','rb')) #the file is load from the local path. In my case from C:/Users/Yoiz Nuñez
        svr_model = ('SVR',svr)
        
        if len(features_list)==1 and features_list[0]=='nw':
           MEC=1
           IAS=0
           ale_model=0
           
        else:
        
            #ALE
            explainer_svr = pymint.InterpretToolkit(svr_model,X=X_train_df, y=y_train)
            ale_model = explainer_svr.ale(features=features_list, n_bins=30, n_jobs=1, subsample=1.0, n_bootstrap=1)
            
        
            MEC = explainer_svr.main_effect_complexity(ale_model,max_segments=10)
        
            IAS= explainer_svr.interaction_strength(ale_model)
    
        return MEC, IAS, ale_model
    
    #SUBSET 1
    MEC_p1, IAS_p1,ale_model = IML(X_train_p1,y_train_p1,X_train_df_p1,features_list_p1)
    
    #values for x and y axis of the plot ALE
    #predictor nw
    ale_nw_value_s1 = 0
    ale_nw_value_s1= 0
    ale_nw_bin_s1 = 0

    
    #SUBSET 2
    MEC_p2, IAS_p2,ale_model = IML(X_train_p2,y_train_p2,X_train_df_p2,features_list_p2)
    MEC_p2= MEC_p2['SVR']
    IAS_p2=IAS_p2.SVR_ias[0] #to get the float value from the dataarray.DataArray result
    IAS_p2 = IAS_p2.to_masked_array()
    IAS_p2=IAS_p2.tolist() 
    
    #values for x and y axis of the plot ALE
    #predictor nw
    ale_nw_value_s2 = ale_model.nw__SVR__ale.to_masked_array()  
    ale_nw_value_s2= ale_nw_value_s2.ravel() #to reduce 1 dim
    ale_nw_bin_s2 = ale_model.nw__bin_values.to_masked_array()
    
    #predictor dy
    ale_dy_value_s2 = ale_model.dy__SVR__ale.to_masked_array()  
    ale_dy_value_s2= ale_dy_value_s2.ravel() #to reduce 1 dim
    ale_dy_bin_s2 = ale_model.dy__bin_values.to_masked_array()
    
    #SUBSET 3
    MEC_p3, IAS_p3,ale_model = IML(X_train_p3,y_train_p3,X_train_df_p3,features_list_p3)
    MEC_p3= MEC_p3['SVR']
    IAS_p3=IAS_p3.SVR_ias[0] #to get the float value from the dataarray.DataArray result
    IAS_p3 = IAS_p3.to_masked_array()
    IAS_p3=IAS_p3.tolist() 
    
    #values for x and y axis of the plot ALE
    #predictor nw
    ale_nw_value_s3 = ale_model.nw__SVR__ale.to_masked_array()  
    ale_nw_value_s3= ale_nw_value_s3.ravel() #to reduce 1 dim
    ale_nw_bin_s3 = ale_model.nw__bin_values.to_masked_array()
    
    #predictor dy
    ale_dy_value_s3 = ale_model.dy__SVR__ale.to_masked_array()  
    ale_dy_value_s3= ale_dy_value_s3.ravel() #to reduce 1 dim
    ale_dy_bin_s3 = ale_model.dy__bin_values.to_masked_array()
    
    #predictor dx
    ale_dx_value_s3 = ale_model.dx__SVR__ale.to_masked_array()  
    ale_dx_value_s3= ale_dx_value_s3.ravel() #to reduce 1 dim
    ale_dx_bin_s3 = ale_model.dx__bin_values.to_masked_array()
    
    
    #subset 4
    MEC_p4, IAS_p4,ale_model = IML(X_train_p4,y_train_p4,X_train_df_p4,features_list_p4)
    MEC_p4= MEC_p4['SVR']
    IAS_p4=IAS_p4.SVR_ias[0] #to get the float value from the dataarray.DataArray result
    IAS_p4 = IAS_p4.to_masked_array()
    IAS_p4=IAS_p4.tolist() 
    
    #values for x and y axis of the plot ALE
    #predictor nw
    ale_nw_value_s4 = ale_model.nw__SVR__ale.to_masked_array()  
    ale_nw_value_s4= ale_nw_value_s4.ravel() #to reduce 1 dim
    ale_nw_bin_s4 = ale_model.nw__bin_values.to_masked_array()
    
    #predictor dx
    ale_dx_value_s4 = ale_model.dx__SVR__ale.to_masked_array()  
    ale_dx_value_s4= ale_dx_value_s4.ravel() #to reduce 1 dim
    ale_dx_bin_s4 = ale_model.dx__bin_values.to_masked_array()
    
    #predictor dy
    ale_dy_value_s4 = ale_model.dy__SVR__ale.to_masked_array()  
    ale_dy_value_s4= ale_dy_value_s4.ravel() #to reduce 1 dim
    ale_dy_bin_s4 = ale_model.dy__bin_values.to_masked_array()
    
    #predictor f
    ale_f_value_s4 = ale_model.f__SVR__ale.to_masked_array()  
    ale_f_value_s4= ale_f_value_s4.ravel() #to reduce 1 dim
    ale_f_bin_s4 = ale_model.f__bin_values.to_masked_array()
    
    #subset 5
    MEC_p5, IAS_p5,ale_model = IML(X_train_p5,y_train_p5,X_train_df_p5,features_list_p5)
    MEC_p5= MEC_p5['SVR']
    IAS_p5=IAS_p5.SVR_ias[0] #to get the float value from the dataarray.DataArray result
    IAS_p5 = IAS_p5.to_masked_array()
    IAS_p5=IAS_p5.tolist() 
    
    #values for x and y axis of the plot ALE
    #predictor nw
    ale_nw_value_s5 = ale_model.nw__SVR__ale.to_masked_array()  
    ale_nw_value_s5= ale_nw_value_s5.ravel() #to reduce 1 dim
    ale_nw_bin_s5 = ale_model.nw__bin_values.to_masked_array()
    
    #predictor dx
    ale_dx_value_s5 = ale_model.dx__SVR__ale.to_masked_array()  
    ale_dx_value_s5= ale_dx_value_s5.ravel() #to reduce 1 dim
    ale_dx_bin_s5 = ale_model.dx__bin_values.to_masked_array()
    
    #predictor dy
    ale_dy_value_s5 = ale_model.dy__SVR__ale.to_masked_array()  
    ale_dy_value_s5= ale_dy_value_s5.ravel() #to reduce 1 dim
    ale_dy_bin_s5 = ale_model.dy__bin_values.to_masked_array()
    
    #predictor f
    ale_f_value_s5 = ale_model.f__SVR__ale.to_masked_array()  
    ale_f_value_s5= ale_f_value_s5.ravel() #to reduce 1 dim
    ale_f_bin_s5 = ale_model.f__bin_values.to_masked_array()
    
    #predictor f
    ale_d_value_s5 = ale_model.d__SVR__ale.to_masked_array()  
    ale_d_value_s5= ale_d_value_s5.ravel() #to reduce 1 dim
    ale_d_bin_s5 = ale_model.d__bin_values.to_masked_array()
    
    ALE_results = {"Subset 1":[ale_nw_value_s1,ale_nw_bin_s1],
                   "Subset 2":[ale_nw_value_s2,ale_nw_bin_s2,ale_dy_value_s2,ale_dy_bin_s2],
                   "Subset 3":[ale_nw_value_s3,ale_nw_bin_s3,ale_dy_value_s3,ale_dy_bin_s3,ale_dx_value_s3,ale_dx_bin_s3],
                   "Subset 4":[ale_nw_value_s4,ale_nw_bin_s4,ale_dy_value_s4,ale_dy_bin_s4,ale_dx_value_s4,ale_dx_bin_s4,ale_f_value_s4,ale_f_bin_s4],
                   "Subset 5":[ale_nw_value_s5,ale_nw_bin_s5,ale_dy_value_s5,ale_dy_bin_s5,ale_dx_value_s5,ale_dx_bin_s5,ale_f_value_s5,ale_f_bin_s5,ale_d_value_s5,ale_d_bin_s5],
                     }
      
    
    IML_results_svr={"Subset 1 predictor":[IAS_p1,MEC_p1],
                    "Subset 2 predictors":[IAS_p2,MEC_p2],
                    "Subset 3 predictors":[IAS_p3,MEC_p3],
                    "Subset 4 predictors":[IAS_p4,MEC_p4],
                    "Subset 5 predictors":[IAS_p5,MEC_p5]
                    }
    
    """
    Calculating PFI for Table 7
    """
    print("----Calculating PFI SVR model---")
    
    def PermImportance(X,y,clf,num_iterations=10):
        
        y_target=y
        y_pred_train=clf.predict(X)
        
        # desnormalization
        y_pred = y_pred_train.reshape(-1,1)
        y_pred = scaler.inverse_transform(y_pred) 
        y_target = scaler.inverse_transform(y) 
        
        
        # rmse
        mse = np.square(np.subtract(y_target,y_pred)).mean()
        baseline_metric = math.sqrt(mse)
        
        scores={c:[] for c in X.columns}
        for c in X.columns:
            print(c)
            X1=X.copy(deep=True)
            for _ in range(num_iterations):
                
                temp=X1[c].tolist()
                random.shuffle(temp)
                X1[c]=temp
                
                y_target=y
                y_pred_train=clf.predict(X1)
                
                # desnormalization
                y_pred = y_pred_train.reshape(-1,1)
                y_pred = scaler.inverse_transform(y_pred) 
                y_target = scaler.inverse_transform(y) 
                
                # rmse
                mse = np.square(np.subtract(y_target,y_pred)).mean()
                score = math.sqrt(mse)
                
                scores[c].append((baseline_metric-score)*-1)
    
        return baseline_metric, scores
    
    
    #subset of 2 predictors
    svr_regressor = svr(X_train_p2,y_train_p2)
    baseline, scores= PermImportance(X_train_df_p2,y_train_p2,svr_regressor,num_iterations=10)
    
    pred1=[]
    pred2=[]
    
    pred1.append(scores['nw'])
    PFI_pred1= pred1
    PFI_pred1_s2=np.mean(PFI_pred1)
    
    pred2.append(scores['dy'])
    PFI_pred2= pred2
    PFI_pred2_s2=np.mean(PFI_pred2)
    
    #subset of 3 predictors
    svr_regressor = svr(X_train_p3,y_train_p3)
    baseline, scores= PermImportance(X_train_df_p3,y_train_p3,svr_regressor,num_iterations=10)
    
    pred1=[]
    pred2=[]
    pred3=[]
    
    pred1.append(scores['nw'])
    PFI_pred1= pred1
    PFI_pred1_s3=np.mean(PFI_pred1)
    
    pred2.append(scores['dy'])
    PFI_pred2= pred2
    PFI_pred2_s3=np.mean(PFI_pred2)
    
    pred3.append(scores['dx'])
    PFI_pred3= pred3
    PFI_pred3_s3=np.mean(PFI_pred3)
    
    #subset of 4 predictors
    svr_regressor = svr(X_train_p4,y_train_p4)
    baseline, scores= PermImportance(X_train_df_p4,y_train_p4,svr_regressor,num_iterations=10)
    
    pred1=[]
    pred2=[]
    pred3=[]
    pred4=[]
    
    pred1.append(scores['nw'])
    PFI_pred1= pred1
    PFI_pred1_s4=np.mean(PFI_pred1)
    
    pred2.append(scores['dy'])
    PFI_pred2= pred2
    PFI_pred2_s4=np.mean(PFI_pred2)
    
    pred3.append(scores['dx'])
    PFI_pred3= pred3
    PFI_pred3_s4=np.mean(PFI_pred3)
    
    pred4.append(scores['f'])
    PFI_pred4= pred4
    PFI_pred4_s4=np.mean(PFI_pred4)
    
    #subset of 5 predictors
    svr_regressor = svr(X_train_p5,y_train_p5)
    baseline, scores= PermImportance(X_train_df_p5,y_train_p5,svr_regressor,num_iterations=10)
    
    pred1=[]
    pred2=[]
    pred3=[]
    pred4=[]
    pred5=[]
    
    pred1.append(scores['nw'])
    PFI_pred1= pred1
    PFI_pred1_s5=np.mean(PFI_pred1)
    
    pred2.append(scores['dy'])
    PFI_pred2= pred2
    PFI_pred2_s5=np.mean(PFI_pred2)
    
    pred3.append(scores['dx'])
    PFI_pred3= pred3
    PFI_pred3_s5=np.mean(PFI_pred3)
    
    pred4.append(scores['f'])
    PFI_pred4= pred4
    PFI_pred4_s5=np.mean(PFI_pred4)
    
    pred5.append(scores['d'])
    PFI_pred5= pred5
    PFI_pred5_s5=np.mean(PFI_pred5)
    
    
    PFI_results_svr={"Subset 2 predictors":[PFI_pred1_s2,PFI_pred2_s2],
                     "Subset 3 predictors":[PFI_pred1_s3,PFI_pred2_s3,PFI_pred3_s3],
                     "Subset 4 predictors":[PFI_pred1_s4,PFI_pred2_s4,PFI_pred3_s4,PFI_pred4_s4],
                     "Subset 5 predictors":[PFI_pred1_s5,PFI_pred2_s5,PFI_pred3_s5,PFI_pred4_s5,PFI_pred5_s5]
                    }
    
    
    """
    Results for Table 9: Performance and complexities of the four regression models having optimized
    predictors subsets and hyperparameters
    """
    print("----Final model with optimized predictors subsets SVR model---")
    
    y_target_train,y_target_test,y_pred_train,y_pred_test = path_loss_svr(X_train_p5,y_train_p5,X_test_p5,y_test_p5)
    RMSE_train, MAPE_train, SD_train, R2_train = functions_tools.statistics(y_target_train, y_pred_train)
    RMSE_test_test, MAPE_test, SD_test, R2_test = functions_tools.statistics(y_target_test, y_pred_test)
    
    
    final_model = {"train":[RMSE_train, MAPE_train, SD_train, R2_train],
                   "test":[RMSE_test_test, MAPE_test, SD_test, R2_test],
                   "IML":[IAS_p5,MEC_p5]
                 }
    
    return svr_results,mi_results_svr,IML_results_svr,PFI_results_svr,ALE_results,final_model


