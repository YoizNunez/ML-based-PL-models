# -*- coding: utf-8 -*-

"""
The code below corresponds to the calculations performed to generate results presented in Section 4 for the Machine Learning models, presented in the following order.
1. Results for Table 4 (Performance evaluation of the Machine Learning models on both the training and testing sets.)
2. Figure 2 (Comparison of measured and predicted path losses on the testing set for the ML models.)
3. Code to generate result of Table 5 (Computational complexity in the ML models).
"""

#functions to apply cross-validation for the hyperparameters setting
from gtb_cv import gtb_model
from rf_cv import rf_model
from svr_cv import svr_model
from mlp_cv import mlp_model

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MultipleLocator
import tabulate
from tabulate import tabulate


def main_section4(df):
        
    """
    Call functions to apply cross-validation in the ML models. The returned results are the best 
    hyperparameters found in each model along with the performance indicators presented in Table 4 and computational complexity
    from Table 5.
    """
    
    print('----Applying Cross-validation method----')
    gtb_results,number_trees_gtb,time_gtb,file_KB_gtb= gtb_model(df) #GTB 
    
    rf_results,number_trees_rf,time_rf,file_KB_rf= rf_model(df) #RF
    
    svr_results,num_sv,time_svr,file_KB_svr= svr_model(df) #SVR
    
    mlp_results,tot_weigths,time_mlp,file_KB_mlp = mlp_model(df) #ANN
    
    #----------------------------------------------------------------------------------
    
    """
    Results of the ML models for Table 4. 
    Performance indicators RMSE, MAPE, standar deviation (SD) and R2 in the training and 
    testing set. 
    """
    
    #ANN
    print('-------------------------------------------------------------')
    data = [
    ['Train',mlp_results['train_mlp'][2],mlp_results['train_mlp'][3],mlp_results['train_mlp'][4],mlp_results['train_mlp'][5]],
    ['Test',mlp_results['test_mlp'][2],mlp_results['test_mlp'][3],mlp_results['test_mlp'][4],mlp_results['test_mlp'][5]]
    ]
    print(tabulate(data, headers=["ANN Model","RMSE [dB]","MAPE [%]","SD [dB]","R2"])) 
    print('-------------------------------------------------------------')
    
    
    #SVR 
    print('-------------------------------------------------------------')
    data = [
    ['Train',svr_results['train_svr'][2],svr_results['train_svr'][3],svr_results['train_svr'][4],svr_results['train_svr'][5]],
    ['Test',svr_results['test_svr'][2],svr_results['test_svr'][3],svr_results['test_svr'][4],svr_results['test_svr'][5]]
    ]
    print(tabulate(data, headers=["SVR Model","RMSE [dB]","MAPE [%]","SD [dB]","R2"]))
    print('-------------------------------------------------------------')
        
    
    #RF 
    print('-------------------------------------------------------------')
    data = [
    ['Train',rf_results['train_rf'][2],rf_results['train_rf'][3],rf_results['train_rf'][4],rf_results['train_rf'][5]],
    ['Test',rf_results['test_rf'][2],rf_results['test_rf'][3],rf_results['test_rf'][4],rf_results['test_rf'][5]]
    ]
    print(tabulate(data, headers=["RF Model","RMSE [dB]","MAPE [%]","SD [dB]","R2"]))
    print('-------------------------------------------------------------')
    
    
    #GTB 
    print('-------------------------------------------------------------')
    data = [
    ['Train',gtb_results['train_gtb'][2],gtb_results['train_gtb'][3],gtb_results['train_gtb'][4],gtb_results['train_gtb'][5]],
    ['Test',gtb_results['test_gtb'][2],gtb_results['test_gtb'][3],gtb_results['test_gtb'][4],gtb_results['test_gtb'][5]]
    ]
    print(tabulate(data, headers=["GTB Model","RMSE [dB]","MAPE [%]","SD [dB]","R2"]))
    print('-------------------------------------------------------------')
    
    
    
    #----------------------------------------------------------------------------------
    """
    Result for Figure 2: Comparison of measured and predicted path losses on the testing set 
    for the ML and empirical models
    """
    
    fig = plt.figure(figsize=(11,5))
    spec = gridspec.GridSpec(ncols=1, nrows=1) 
    
    ax1 = fig.add_subplot(spec[0,0]) 
    
    p1 = max(max(mlp_results['test_mlp'][0]), max(mlp_results['test_mlp'][0]))
    p2 = min(min(mlp_results['test_mlp'][0]), min(mlp_results['test_mlp'][0]))
    ax1.plot([p1, p2], [p1, p2], '--',linewidth=1,label= 'Best fit',c='navy')
    ax1.set_xlabel('Measured path-loss [dB]', fontsize=16)
    ax1.set_ylabel('Predicted path-loss [dB]', fontsize=16)
    
    #subplot for the ML models
    #y_target, y_pred
    ax1.scatter(mlp_results['test_mlp'][0],mlp_results['test_mlp'][1],s=75, label= 'ANN, $R^2 = 0.89$',c='olivedrab',marker='+')
    ax1.scatter(svr_results['test_svr'][0], svr_results['test_svr'][1],s=55, label= 'SVR, $R^2 = 0.88$',c='darksalmon',marker='D',linewidths=0)
    ax1.scatter(rf_results['test_rf'][0], rf_results['test_rf'][1],s=55, label= 'RF,  $R^2 = 0.88$',c='turquoise')
    ax1.scatter(gtb_results['test_gtb'][0], gtb_results['test_gtb'][1],s=55,label= 'GTB, $R^2 = 0.92$',c='crimson',marker='*')
    ax1.legend(loc='lower right',fontsize=11)
    ax1.grid(ls='--')
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.set_xlim((70,140))
    ax1.xaxis.set_major_locator(MultipleLocator(10))
    
    plt.figtext(0.3, 0.91, '(a) ML based models', ha='center', va='center',fontsize='16')
    plt.subplots_adjust(wspace=0.3)
    
    #plt.savefig('R2.eps',format='eps',dpi=1200)
    plt.show()
    plt.close()
    
    
    #----------------------------------------------------------------------------------
    """
    Results for Table 5: Computational complexity
    """
    print('----Results for Table 5: Computational complexity----')
    data = [
    ['ANN',time_mlp,file_KB_mlp,tot_weigths],
    ['SVR',time_svr,file_KB_svr,num_sv[0]],
    ['RF',time_rf,file_KB_rf,number_trees_rf],
    ['GTB',time_gtb,file_KB_gtb,number_trees_gtb]
    ]
    
    print(tabulate(data, headers=["Model","Training time [s]","Memory storage [KB]","No. of parameters"]))
    print('-------------------------------------------------------------')

    return print("Finished Section 4")