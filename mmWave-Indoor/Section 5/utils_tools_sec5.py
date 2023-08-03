# -*- coding: utf-8 -*-

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#libraries for data preprocessing 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#libraries for statistical measures
from sklearn.metrics import mean_squared_error, r2_score
import math


"""
Functions to normalize data
"""
def normalize_data(X,y):

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # normalize inputs
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

  # convert output variable to float
  y_train, y_test = y_train.astype(float), y_test.astype(float),

  return X_train, X_test, y_train, y_test,scaler


def normalize_data_input(X):

  X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

  # normalize inputs
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  X_train = np.array(X_train)
  X_test = np.array(X_test)

  return X_train, X_test,scaler

def normalize_data_output(y):

  y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)

  # normalize output
  scaler = StandardScaler()
  y_train =  np.array(y_train)
  y_test =  np.array(y_test)

  y_train = y_train.reshape(-1, 1)
  y_train = scaler.fit_transform(y_train)
  y_test = y_test.reshape(-1, 1)
  y_test = scaler.transform(y_test)

  # convert output variable to float
  y_train, y_test = y_train.astype(float), y_test.astype(float),

  return y_train, y_test,scaler

class functions_tools():
   
    """
    Functions to calculate performance indicators
    """
    
    def statistics(y_target,y_pred):
    
      # rmse
      mse = np.square(np.subtract(y_target,y_pred)).mean()
      rmse = math.sqrt(mse)
      
      #mape
      mape = np.mean(np.abs((y_target - y_pred) / y_target)) * 100
    
      # sd
      n = len(y_pred)
      sum_model = 0 
    
      abs_dif = np.abs(y_target-y_pred)
      mean = np.mean(abs_dif)
    
      for x in abs_dif:
          t = (x - mean) ** 2
          sum_model += t 
    
      sd = math.sqrt(sum_model/(n))
    
      # r2
      r2= r2_score(y_target, y_pred) 
    
      return rmse, mape, sd, r2
    
    
    """
    Functions to normalize each subset of predictors
    """
    
    # function to normalize data
    def correct_normalize_one(pred1_eval,df,y):
      
      if pred1_eval==4:
    
        # normalize categorical input
        X_nw = df.iloc[:, [4]]
        X_train, X_test = train_test_split(X_nw, test_size=0.2, random_state=42)
                
    
        y_train, y_test,scaler= normalize_data_output(y)
    
      else: 
             
        num_list = [pred1_eval] # create a list to choose only numerical input
        X = df.iloc[:, num_list] # Final dataset without categorical input  
        X_train, X_test, y_train, y_test,scaler= normalize_data(X,y)
        
      return X_train, X_test, y_train, y_test
    
    
    def correct_normalize_two(pred1, pred2_eval,df,y):
      
        if pred1==4:
            
            # normalize categorical input
            X_nw = df.iloc[:, [4]]
            X_train_pred1, X_test_pred1 = train_test_split(X_nw, test_size=0.2, random_state=42)
                    
            
        else:
            
            X = df.iloc[:, [pred1]]
            X_train_pred1,X_test_pred1,scaler= normalize_data_input(X) # call the function to normalize compute
            
        
        if pred2_eval==4:
            
            # normalize categorical input
            X_nw = df.iloc[:, [4]]
            X_train_pred2, X_test_pred2 = train_test_split(X_nw, test_size=0.2, random_state=42)
                    
            
        else:
            
            X = df.iloc[:, [pred2_eval]]
            X_train_pred2, X_test_pred2,scaler= normalize_data_input(X) # call the function to normalize compute
            
        X_train= np.concatenate([X_train_pred1,X_train_pred2],axis=1)
        X_test= np.concatenate([X_test_pred1,X_test_pred2],axis=1)
    
        y_train, y_test,scaler= normalize_data_output(y)
        
        return X_train, X_test, y_train, y_test
    
    def correct_normalize_three(pred1,pred2,pred3_eval,df,y):
      
        if pred1==4:
            
            # normalize categorical input
            X_nw = df.iloc[:, [4]]
            X_train_pred1, X_test_pred1 = train_test_split(X_nw, test_size=0.2, random_state=42)
                    
            
        else:
            
            X = df.iloc[:, [pred1]]
            X_train_pred1,X_test_pred1,scaler= normalize_data_input(X) # call the function to normalize compute
            
        if pred2==4:
            
            # normalize categorical input
            X_nw = df.iloc[:, [4]]
            X_train_pred2, X_test_pred2 = train_test_split(X_nw, test_size=0.2, random_state=42)
                    
            
        else:
            
            X = df.iloc[:, [pred2]]
            X_train_pred2,X_test_pred2,scaler= normalize_data_input(X) # call the function to normalize compute
        
        
        if pred3_eval==4:
            
            # normalize categorical input
            X_nw = df.iloc[:, [4]]
            X_train_pred3, X_test_pred3 = train_test_split(X_nw, test_size=0.2, random_state=42)
                    
            
        else:
            
            X = df.iloc[:, [pred3_eval]]
            X_train_pred3, X_test_pred3,scaler= normalize_data_input(X) # call the function to normalize compute
            
        X_train= np.concatenate([X_train_pred1,X_train_pred2,X_train_pred3],axis=1)
        X_test= np.concatenate([X_test_pred1,X_test_pred2,X_test_pred3],axis=1)
    
        y_train, y_test,scaler= normalize_data_output(y)
        
        return X_train, X_test, y_train, y_test
    
    
    def correct_normalize_four(pred1,pred2,pred3,pred4_eval,df,y):
      
        if pred1==4:
            
            # normalize categorical input
            X_nw = df.iloc[:, [4]]
            X_train_pred1, X_test_pred1 = train_test_split(X_nw, test_size=0.2, random_state=42)
    
            
        else:
            
            X = df.iloc[:, [pred1]]
            X_train_pred1,X_test_pred1,scaler= normalize_data_input(X) # call the function to normalize compute
            
        if pred2==4:
            
            # normalize categorical input
            X_nw = df.iloc[:, [4]]
            X_train_pred2, X_test_pred2 = train_test_split(X_nw, test_size=0.2, random_state=42)
                    
            
        else:
            
            X = df.iloc[:, [pred2]]
            X_train_pred2,X_test_pred2,scaler= normalize_data_input(X) # call the function to normalize compute
        
        if pred3==4:
            
            # normalize categorical input
            X_nw = df.iloc[:, [4]]
            X_train_pred3, X_test_pred3 = train_test_split(X_nw, test_size=0.2, random_state=42)
                    
            
        else:
            
            X = df.iloc[:, [pred3]]
            X_train_pred3,X_test_pred3,scaler= normalize_data_input(X) # call the function to normalize compute
        
        if pred4_eval==4:
            
            # normalize categorical input
            X_nw = df.iloc[:, [4]]
            X_train_pred4, X_test_pred4 = train_test_split(X_nw, test_size=0.2, random_state=42)
    
        else:
            
            X = df.iloc[:, [pred4_eval]]
            X_train_pred4, X_test_pred4,scaler= normalize_data_input(X) # call the function to normalize compute
                
        
        X_train= np.concatenate([X_train_pred1,X_train_pred2,X_train_pred3,X_train_pred4],axis=1)
        X_test= np.concatenate([X_test_pred1,X_test_pred2,X_test_pred3,X_test_pred4],axis=1)
    
        y_train, y_test,scaler= normalize_data_output(y)
        
        return X_train, X_test, y_train, y_test
    
    
    def correct_normalize_five(pred1,pred2,pred3,pred4,pred5_eval,df,y):
      
         if pred1==4:
            
            # normalize categorical input
            X_nw = df.iloc[:, [4]]
            X_train_pred1, X_test_pred1 = train_test_split(X_nw, test_size=0.2, random_state=42)
                    
            
         else:
            
            X = df.iloc[:, [pred1]]
            X_train_pred1,X_test_pred1,scaler= normalize_data_input(X) # call the function to normalize compute
            
         if pred2==4:
            
            # normalize categorical input
            X_nw = df.iloc[:, [4]]
            X_train_pred2, X_test_pred2 = train_test_split(X_nw, test_size=0.2, random_state=42)
    
            
         else:
            
            X = df.iloc[:, [pred2]]
            X_train_pred2,X_test_pred2,scaler= normalize_data_input(X) # call the function to normalize compute
        
         if pred3==4:
            
            # normalize categorical input
            X_nw = df.iloc[:, [4]]
            X_train_pred3, X_test_pred3 = train_test_split(X_nw, test_size=0.2, random_state=42)
                    
            
         else:
            
            X = df.iloc[:, [pred3]]
            X_train_pred3,X_test_pred3,scaler= normalize_data_input(X) # call the function to normalize compute
        
         if pred4==4:
            
            # normalize categorical input
            X_nw = df.iloc[:, [4]]
            X_train_pred4, X_test_pred4 = train_test_split(X_nw, test_size=0.2, random_state=42)
                    
            
         else:
            
            X = df.iloc[:, [pred4]]
            X_train_pred4,X_test_pred4,scaler= normalize_data_input(X) # call the function to normalize compute
        
         if pred5_eval==4:
            
            # normalize categorical input
            X_nw = df.iloc[:, [4]]
            X_train_pred5, X_test_pred5 = train_test_split(X_nw, test_size=0.2, random_state=42)
                            
         else:
            
            X = df.iloc[:, [pred5_eval]]
            X_train_pred5, X_test_pred5,scaler= normalize_data_input(X) # call the function to normalize compute
                
        
         X_train= np.concatenate([X_train_pred1,X_train_pred2,X_train_pred3,X_train_pred4,X_train_pred5],axis=1)
         X_test= np.concatenate([X_test_pred1,X_test_pred2,X_test_pred3,X_test_pred4,X_test_pred5],axis=1)
    
         y_train, y_test,scaler= normalize_data_output(y)
        
         return X_train, X_test, y_train, y_test
