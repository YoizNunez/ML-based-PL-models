# -*- coding: utf-8 -*-

#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import math
from sklearn.metrics import mean_squared_error, r2_score

def performance_indicators(y_target,y_pred):
    
    #RMSE 
    MSE = np.square(np.subtract(y_target,y_pred)).mean()
    RMSE = math.sqrt(MSE)
    
    #R2
    R2 = r2_score(y_target, y_pred) 
    
    #SD 
    n = len(y_pred)
    sum_model=0
    
    abs_dif = np.abs(y_target-y_pred)
    mean_model = np.mean(abs_dif)
    
    for x in abs_dif:
        t = (x - mean_model) ** 2
        sum_model += t 
    
    SD = math.sqrt(sum_model/(n))
    
    #MAPE 
    MAPE = np.mean(np.abs((y_target - y_pred) / y_target)) * 100
    
    return RMSE, MAPE, SD, R2