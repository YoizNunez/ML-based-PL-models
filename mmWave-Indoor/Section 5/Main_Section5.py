# -*- coding: utf-8 -*-

"""
The code below corresponds to the calculations performed to generate results presented in Section 5 for the Machine Learning models, presented in the following order.

1. Figure 3: Scatter plots between predictors and the path-loss and between predictors pairs..
2. Results for Table 6: Performance indicators, Mutual information and IML indicators for each ML model.
3. Results for Table 7: PFI rank for each subset evaluated in each model.
4. Figure 4: ALE Graphs for the four ML models.
5. Results for Table 8: Interaction strength between two predictors (2D-ALE) for the GTB-based PL
model.
6. Figure 5: Marginal contributions in performance and interpretability for the
GTB-based PL model.
7. Results for Table 9: Performances and complexities of the four final regression having optimized
predictors subsets and hyperparameters.
8. Figure 6: Measured and predicted path-loss versus distance and the number of traversed walls for the GTB model on the testing set.
9. Results for Table 10: Performance of the proposed GTB-based PL model when the models are training considering a set of transmitters and the test employs a different one.
10. Results for Table 10: Generalization capability analysis for the GTB-based PL model. The models are trained using two transmitters and evaluated (tested) on the four remaining ones.
11. Results for Table 12: GTB and empirical PL models performance using the database from Yonsei University, Korea.
"""

import pandas as pd

from mutual_information import mutual_information_plot
from ann_model import ann_model_IML
from svr_model import svr_model_IML
from rf_model import rf_model_IML
from gtb_model import gtb_model_IML

from GTB_Yonsei import gtb_yonsei 
from generability_GTB_model import generalization_gtb_model
from generability_Tx_GTB_model import generalization_gtb_model_tx

import tabulate
from tabulate import tabulate
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec


def main_section5(df,df_tx,df_korea):
      
    """
    Call function to plot Figure 3
    """
    mutual_information_plot(df)
    
    """
    Call main functions for each model
    """
    ann_results, mi_results_ann, IML_results_ann, PFI_results_ann,ALE_results_ann,final_model_ann = ann_model_IML(df)
    
    svr_results, mi_results_svr, IML_results_svr,PFI_results_svr,ALE_results_svr,final_model_svr= svr_model_IML(df)
    
    rf_results, mi_results_rf, IML_results_rf,PFI_results_rf,ALE_results_rf,final_model_rf= rf_model_IML(df)
    
    gtb_results, mi_results_gtb, IML_results_gtb,PFI_results_gtb,INT_2D_results, ALE_results_gtb,final_model_gtb,matrix_27GHz,matrix_33GHz,matrix_40GHz = gtb_model_IML(df)
    
    
    """
    Code for Table 10
    """
    results_generalization_gtb = generalization_gtb_model(df_tx)
    
    """
    Code for Table 11
    """
    results_generalization_gtb_tx = generalization_gtb_model_tx(df_tx)
    
    """
    Code for Table 12
    """
    results_yonsei= gtb_yonsei(df_korea)
  
    #------------------------------------------------------------------------------------------------------
    """
    Results of the ML models for Table 6. 
    Performance indicators RMSE, MAPE, standar deviation (SD) and R2 in the training set, Mutual information (JMI and CMI), 
    and performance IML. 
    """
    print('---------------Table 6--------------------')
    #ANN
    print('-------------------------------------------------------------')
    #performance indicators
    data = [
    ["Subset 1",ann_results['Pred1'][1],ann_results['Pred1'][2],ann_results['Pred1'][3],ann_results['Pred1'][4]],
    ["Subset 2",ann_results['Pred2'][1],ann_results['Pred2'][2],ann_results['Pred2'][3],ann_results['Pred2'][4]],
    ["Subset 3",ann_results['Pred3'][1],ann_results['Pred3'][2],ann_results['Pred3'][3],ann_results['Pred3'][4]],
    ["Subset 4",ann_results['Pred4'][1],ann_results['Pred4'][2],ann_results['Pred4'][3],ann_results['Pred4'][4]],
    ["Subset 5",ann_results['Pred5'][1],ann_results['Pred5'][2],ann_results['Pred5'][3],ann_results['Pred5'][4]]
    ]
    print(tabulate(data, headers=["Subsets (ANN)","RMSE [dB]","MAPE [%]","SD [dB]","R2"])) 
    
    #mutual information
    data = [
    ["Subset 2",mi_results_ann['Subset 2 predictors'][0],mi_results_ann['Subset 2 predictors'][1]],
    ["Subset 3",mi_results_ann['Subset 3 predictors'][0],mi_results_ann['Subset 3 predictors'][1]],
    ["Subset 4",mi_results_ann['Subset 4 predictors'][0],mi_results_ann['Subset 4 predictors'][1]],
    ["Subset 5",mi_results_ann['Subset 5 predictors'][0],mi_results_ann['Subset 5 predictors'][1]]
    ]
    print(tabulate(data, headers=["Subsets (ANN)","JMI","CMI"])) 
    
    #IML indicators
    data = [
    ["Subset 1",'-',IML_results_ann['Subset 1 predictor'][1]],
    ["Subset 2",IML_results_ann['Subset 2 predictors'][0],IML_results_ann['Subset 2 predictors'][1]],
    ["Subset 3",IML_results_ann['Subset 3 predictors'][0],IML_results_ann['Subset 3 predictors'][1]],
    ["Subset 4",IML_results_ann['Subset 4 predictors'][0],IML_results_ann['Subset 4 predictors'][1]],
    ["Subset 5",IML_results_ann['Subset 5 predictors'][0],IML_results_ann['Subset 5 predictors'][1]]
    ]
    print(tabulate(data, headers=["Subsets (ANN)","IAS","MEC"])) 
    print('-------------------------------------------------------------')
    
    
    #SVR
    print('-------------------------------------------------------------')
    #performance indicators
    data = [
    ["Subset 1",svr_results['Pred1'][1],svr_results['Pred1'][2],svr_results['Pred1'][3],svr_results['Pred1'][4]],
    ["Subset 2",svr_results['Pred2'][1],svr_results['Pred2'][2],svr_results['Pred2'][3],svr_results['Pred2'][4]],
    ["Subset 3",svr_results['Pred3'][1],svr_results['Pred3'][2],svr_results['Pred3'][3],svr_results['Pred3'][4]],
    ["Subset 4",svr_results['Pred4'][1],svr_results['Pred4'][2],svr_results['Pred4'][3],svr_results['Pred4'][4]],
    ["Subset 5",svr_results['Pred5'][1],svr_results['Pred5'][2],svr_results['Pred5'][3],svr_results['Pred5'][4]]
    ]
    print(tabulate(data, headers=["Subsets (SVR)","RMSE [dB]","MAPE [%]","SD [dB]","R2"])) 
    
    #mutual information
    data = [
    ["Subset 2",mi_results_svr['Subset 2 predictors'][0],mi_results_svr['Subset 2 predictors'][1]],
    ["Subset 3",mi_results_svr['Subset 3 predictors'][0],mi_results_svr['Subset 3 predictors'][1]],
    ["Subset 4",mi_results_svr['Subset 4 predictors'][0],mi_results_svr['Subset 4 predictors'][1]],
    ["Subset 5",mi_results_svr['Subset 5 predictors'][0],mi_results_svr['Subset 5 predictors'][1]]
    ]
    print(tabulate(data, headers=["Subsets (SVR)","JMI","CMI"])) 
    
    #IML indicators
    data = [
    ["Subset 1",'-',IML_results_svr['Subset 1 predictor'][1]],
    ["Subset 2",IML_results_svr['Subset 2 predictors'][0],IML_results_svr['Subset 2 predictors'][1]],
    ["Subset 3",IML_results_svr['Subset 3 predictors'][0],IML_results_svr['Subset 3 predictors'][1]],
    ["Subset 4",IML_results_svr['Subset 4 predictors'][0],IML_results_svr['Subset 4 predictors'][1]],
    ["Subset 5",IML_results_svr['Subset 5 predictors'][0],IML_results_svr['Subset 5 predictors'][1]]
    ]
    print(tabulate(data, headers=["Subsets (SVR)","IAS","MEC"])) 
    print('-------------------------------------------------------------')
    
    #RF
    print('-------------------------------------------------------------')
    #performance indicators
    data = [
    ["Subset 1",rf_results['Pred1'][1],rf_results['Pred1'][2],rf_results['Pred1'][3],rf_results['Pred1'][4]],
    ["Subset 2",rf_results['Pred2'][1],rf_results['Pred2'][2],rf_results['Pred2'][3],rf_results['Pred2'][4]],
    ["Subset 3",rf_results['Pred3'][1],rf_results['Pred3'][2],rf_results['Pred3'][3],rf_results['Pred3'][4]],
    ["Subset 4",rf_results['Pred4'][1],rf_results['Pred4'][2],rf_results['Pred4'][3],rf_results['Pred4'][4]],
    ["Subset 5",rf_results['Pred5'][1],rf_results['Pred5'][2],rf_results['Pred5'][3],rf_results['Pred5'][4]]
    ]
    print(tabulate(data, headers=["Subsets (RF)","RMSE [dB]","MAPE [%]","SD [dB]","R2"])) 
    
    #mutual information
    data = [
    ["Subset 2",mi_results_rf['Subset 2 predictors'][0],mi_results_rf['Subset 2 predictors'][1]],
    ["Subset 3",mi_results_rf['Subset 3 predictors'][0],mi_results_rf['Subset 3 predictors'][1]],
    ["Subset 4",mi_results_rf['Subset 4 predictors'][0],mi_results_rf['Subset 4 predictors'][1]],
    ["Subset 5",mi_results_rf['Subset 5 predictors'][0],mi_results_rf['Subset 5 predictors'][1]]
    ]
    print(tabulate(data, headers=["Subsets (RF)","JMI","CMI"])) 
    
    #IML indicators
    data = [
    ["Subset 1",'-',IML_results_rf['Subset 1 predictor'][1]],
    ["Subset 2",IML_results_rf['Subset 2 predictors'][0],IML_results_rf['Subset 2 predictors'][1]],
    ["Subset 3",IML_results_rf['Subset 3 predictors'][0],IML_results_rf['Subset 3 predictors'][1]],
    ["Subset 4",IML_results_rf['Subset 4 predictors'][0],IML_results_rf['Subset 4 predictors'][1]],
    ["Subset 5",IML_results_rf['Subset 5 predictors'][0],IML_results_rf['Subset 5 predictors'][1]]
    ]
    print(tabulate(data, headers=["Subsets (RF)","IAS","MEC"])) 
    print('-------------------------------------------------------------')
    
    
    #GTB
    print('-------------------------------------------------------------')
    #performance indicators
    data = [
    ["Subset 1",gtb_results['Pred1'][1],gtb_results['Pred1'][2],gtb_results['Pred1'][3],gtb_results['Pred1'][4]],
    ["Subset 2",gtb_results['Pred2'][1],gtb_results['Pred2'][2],gtb_results['Pred2'][3],gtb_results['Pred2'][4]],
    ["Subset 3",gtb_results['Pred3'][1],gtb_results['Pred3'][2],gtb_results['Pred3'][3],gtb_results['Pred3'][4]],
    ["Subset 4",gtb_results['Pred4'][1],gtb_results['Pred4'][2],gtb_results['Pred4'][3],gtb_results['Pred4'][4]],
    ["Subset 5",gtb_results['Pred5'][1],gtb_results['Pred5'][2],gtb_results['Pred5'][3],gtb_results['Pred5'][4]]
    ]
    print(tabulate(data, headers=["Subsets (GTB)","RMSE [dB]","MAPE [%]","SD [dB]","R2"])) 
    
    #mutual information
    data = [
    ["Subset 2",mi_results_gtb['Subset 2 predictors'][0],mi_results_gtb['Subset 2 predictors'][1]],
    ["Subset 3",mi_results_gtb['Subset 3 predictors'][0],mi_results_gtb['Subset 3 predictors'][1]],
    ["Subset 4",mi_results_gtb['Subset 4 predictors'][0],mi_results_gtb['Subset 4 predictors'][1]],
    ["Subset 5",mi_results_gtb['Subset 5 predictors'][0],mi_results_gtb['Subset 5 predictors'][1]]
    ]
    print(tabulate(data, headers=["Subsets (GTB)","JMI","CMI"])) 
    
    #IML indicators
    data = [
    ["Subset 1",'-',IML_results_gtb['Subset 1 predictor'][1]],
    ["Subset 2",IML_results_gtb['Subset 2 predictors'][0],IML_results_gtb['Subset 2 predictors'][1]],
    ["Subset 3",IML_results_gtb['Subset 3 predictors'][0],IML_results_gtb['Subset 3 predictors'][1]],
    ["Subset 4",IML_results_gtb['Subset 4 predictors'][0],IML_results_gtb['Subset 4 predictors'][1]],
    ["Subset 5",IML_results_gtb['Subset 5 predictors'][0],IML_results_gtb['Subset 5 predictors'][1]]
    ]
    print(tabulate(data, headers=["Subsets (GTB)","IAS","MEC"])) 
    print('-------------------------------------------------------------')
    
    
    #------------------------------------------------------------------------------------------------------
    """
    Result for the Figure 5: Marginal contributions in performance and interpretability for the
    GTB-based PL model
    """
    print('---------------Figure 5--------------------')
    rmse = [gtb_results['Pred1'][1],gtb_results['Pred1'][1]-gtb_results['Pred2'][1],
                                    gtb_results['Pred2'][1]-gtb_results['Pred3'][1],
                                    gtb_results['Pred3'][1]-gtb_results['Pred4'][1],
                                    gtb_results['Pred4'][1]-gtb_results['Pred5'][1]
                                    ]
    
    ias =  [None,IML_results_gtb['Subset 2 predictors'][0], 
                 IML_results_gtb['Subset 2 predictors'][0]-IML_results_gtb['Subset 3 predictors'][0],
                 IML_results_gtb['Subset 3 predictors'][0]-IML_results_gtb['Subset 4 predictors'][0],
                 IML_results_gtb['Subset 4 predictors'][0]-IML_results_gtb['Subset 5 predictors'][0]
                 ]
        
    mec =  [IML_results_gtb['Subset 1 predictor'][1],
                 IML_results_gtb['Subset 1 predictor'][1]-IML_results_gtb['Subset 2 predictors'][1],
                 IML_results_gtb['Subset 2 predictors'][1]-IML_results_gtb['Subset 3 predictors'][1],
                 IML_results_gtb['Subset 3 predictors'][1]-IML_results_gtb['Subset 4 predictors'][1],
                 IML_results_gtb['Subset 4 predictors'][1]-IML_results_gtb['Subset 5 predictors'][1]
                 ]
    
    x=[1,2,3,4,5]
    
    fig,ax1 = plt.subplots(figsize=(6,5))
    
    ax1.set_ylim(-2,11)
    
    ax1.plot(x,rmse,"-o",lw=3,label ='$\Delta RMSE_p$ [dB]',markersize=8)
    ax1.plot(x,ias,"-o",lw=3,label ='$\Delta IAS_p$',markersize=8)
    ax1.plot(x,mec,"-o",lw=3,label ='$\Delta\overline{MEC}_p$',markersize=8)
    
    ax1.set_xlabel('Predictor subsets',fontsize=13)
    ax1.set_ylabel('Marginal contribution',fontsize=13)
    ax1.yaxis.set_tick_params(labelsize=13)
    ax1.xaxis.set_tick_params(labelsize=13)
    
    ax1.legend(fontsize=13)
    ax1.grid(linestyle='--')
    ax1.set_xlim(0.5,5.5)
    
    plt.savefig('Marg_contribution.eps',format='eps',dpi=1200)
    plt.show()
    plt.close()


    #------------------------------------------------------------------------------------------------------
    """
    Results for Table 7:PFI rank of the selected predictor with RMSE loss for
    different subsets of predictors.
    """
    print('---------------Table 7--------------------')
    #ANN
    data = [
    ["Subset 2",PFI_results_ann['Subset 2 predictors'][0],PFI_results_ann['Subset 2 predictors'][1]], #PFI values for dx and dy, respectively
    ["Subset 3",PFI_results_ann['Subset 3 predictors'][0],PFI_results_ann['Subset 3 predictors'][1],PFI_results_ann['Subset 3 predictors'][2]],#PFI values for dx, dy and f, respectively
    ["Subset 4",PFI_results_ann['Subset 4 predictors'][0],PFI_results_ann['Subset 4 predictors'][1],PFI_results_ann['Subset 4 predictors'][2],PFI_results_ann['Subset 4 predictors'][3]], #PFI values for dx, dy,f and nw, respectively
    ["Subset 5",PFI_results_ann['Subset 5 predictors'][0],PFI_results_ann['Subset 5 predictors'][1],PFI_results_ann['Subset 5 predictors'][2],PFI_results_ann['Subset 5 predictors'][3],PFI_results_ann['Subset 5 predictors'][4]] #PFI values for dx, dy,f,nw and d respectively
    ]
    
    print(tabulate(data, headers=["PFI (loss: RMSE [dB]): ANN model","","","",""])) 
    print('-------------------------------------------------------------')
    
    
    #SVR
    data = [
    ["Subset 2",PFI_results_svr['Subset 2 predictors'][0],PFI_results_svr['Subset 2 predictors'][1]], #PFI values for nw and dy, respectively
    ["Subset 3",PFI_results_svr['Subset 3 predictors'][0],PFI_results_svr['Subset 3 predictors'][1],PFI_results_svr['Subset 3 predictors'][2]],#PFI values for nw, dy and dx, respectively
    ["Subset 4",PFI_results_svr['Subset 4 predictors'][0],PFI_results_svr['Subset 4 predictors'][1],PFI_results_svr['Subset 4 predictors'][2],PFI_results_svr['Subset 4 predictors'][3]], #PFI values for nw, dy,dx and f, respectively
    ["Subset 5",PFI_results_svr['Subset 5 predictors'][0],PFI_results_svr['Subset 5 predictors'][1],PFI_results_svr['Subset 5 predictors'][2],PFI_results_svr['Subset 5 predictors'][3],PFI_results_svr['Subset 5 predictors'][4]] #PFI values for nw,dy,dx,f and d respectively
    ]
    
    print(tabulate(data, headers=["PFI (loss: RMSE [dB]): SVR model","","","",""])) 
    print('-------------------------------------------------------------')
    
    #RF
    data = [
    ["Subset 2",PFI_results_rf['Subset 2 predictors'][0],PFI_results_rf['Subset 2 predictors'][1]], #PFI values for d and nw, respectively
    ["Subset 3",PFI_results_rf['Subset 3 predictors'][0],PFI_results_rf['Subset 3 predictors'][1],PFI_results_rf['Subset 3 predictors'][2]],#PFI values for d, nw and f, respectively
    ["Subset 4",PFI_results_rf['Subset 4 predictors'][0],PFI_results_rf['Subset 4 predictors'][1],PFI_results_rf['Subset 4 predictors'][2],PFI_results_rf['Subset 4 predictors'][3]], #PFI values for d, nw,f and dy, respectively
    ["Subset 5",PFI_results_rf['Subset 5 predictors'][0],PFI_results_rf['Subset 5 predictors'][1],PFI_results_rf['Subset 5 predictors'][2],PFI_results_rf['Subset 5 predictors'][3],PFI_results_rf['Subset 5 predictors'][4]] #PFI values for d,nw,f,dy and dx respectively
    ]
    
    print(tabulate(data, headers=["PFI (loss: RMSE [dB]): RF model","","","",""])) 
    print('-------------------------------------------------------------')
    
    #GTB
    data = [
    ["Subset 2",PFI_results_gtb['Subset 2 predictors'][0],PFI_results_gtb['Subset 2 predictors'][1]], #PFI values for d and f, respectively
    ["Subset 3",PFI_results_gtb['Subset 3 predictors'][0],PFI_results_gtb['Subset 3 predictors'][1],PFI_results_gtb['Subset 3 predictors'][2]],#PFI values for d, f and nw, respectively
    ["Subset 4",PFI_results_gtb['Subset 4 predictors'][0],PFI_results_gtb['Subset 4 predictors'][1],PFI_results_gtb['Subset 4 predictors'][2],PFI_results_gtb['Subset 4 predictors'][3]], #PFI values for d,f,nw and dy, respectively
    ["Subset 5",PFI_results_gtb['Subset 5 predictors'][0],PFI_results_gtb['Subset 5 predictors'][1],PFI_results_gtb['Subset 5 predictors'][2],PFI_results_gtb['Subset 5 predictors'][3],PFI_results_gtb['Subset 5 predictors'][4]] #PFI values for d,f,nw,dy and dx respectively
    ]
    
    print(tabulate(data, headers=["PFI (loss: RMSE [dB]): GTB model","","","",""])) 
    print('-------------------------------------------------------------')


    """
    Results for Table 8: Interaction strength between two predictors (2D-ALE) for the GTB-based PL
    model
    """
    print('---------------Table 8--------------------')
    data = [
    ["Subset 2 predictors",INT_2D_results['Subset 2 predictors'][0]], #values for the interaction between d,f
    ["Subset 3 predictors",INT_2D_results['Subset 3 predictors'][0],INT_2D_results['Subset 3 predictors'][1],INT_2D_results['Subset 3 predictors'][2]],#values for the interaction between f,d and nw
    ["Subset 4 predictors",INT_2D_results['Subset 4 predictors'][0],INT_2D_results['Subset 4 predictors'][1],INT_2D_results['Subset 4 predictors'][2],INT_2D_results['Subset 4 predictors'][3],INT_2D_results['Subset 4 predictors'][4],INT_2D_results['Subset 4 predictors'][5]], #values for the interaction between f,d,nw and dy
    ["Subset 5 predictors",INT_2D_results['Subset 5 predictors'][0],INT_2D_results['Subset 5 predictors'][1],INT_2D_results['Subset 5 predictors'][2],INT_2D_results['Subset 5 predictors'][3],INT_2D_results['Subset 5 predictors'][4],INT_2D_results['Subset 5 predictors'][5],INT_2D_results['Subset 5 predictors'][6],INT_2D_results['Subset 5 predictors'][7],INT_2D_results['Subset 5 predictors'][8],INT_2D_results['Subset 5 predictors'][9]] #PFI values for d,f,nw,dy and dx respectively
    ]
    
    print(tabulate(data, headers=["Interaction 2D -ALE: GTB model","","","","",'','','','','',''])) 
    print('-------------------------------------------------------------')
  
    
    #------------------------------------------------------------------------------------------------------
    """
    Figure 4: ALE graphs for the four ML models
    """
    print('---------------Figure 4--------------------')
    fig = plt.figure(figsize=(15,13))
    spec = gridspec.GridSpec(ncols=4, nrows=5) # 6 columns evenly divides both 2 & 3
    
    ax1 = fig.add_subplot(spec[0,0]) # row 0 with axes spanning 2 cols on evens
    ax2 = fig.add_subplot(spec[1,0]) # row 0 with axes spanning 2 cols on evens
    ax3 = fig.add_subplot(spec[2,0]) # row 0 with axes spanning 2 cols on evens
    ax4 = fig.add_subplot(spec[3,0]) # row 0 with axes spanning 2 cols on evens
    ax5 = fig.add_subplot(spec[4,0]) # row 0 with axes spanning 2 cols on evens
    
    #ANN
    #dx
    ax1.plot(ALE_results_ann['Subset 1'][1],ALE_results_ann['Subset 1'][0],"+-", label ='1',lw=3,color='blue')
    ax1.plot(ALE_results_ann['Subset 2'][1],ALE_results_ann['Subset 2'][0],"-.", label ='2',lw=3,color='red')
    ax1.plot(ALE_results_ann['Subset 3'][1],ALE_results_ann['Subset 3'][0],":", label ='3',lw=3,color='green')
    ax1.plot(ALE_results_ann['Subset 4'][1],ALE_results_ann['Subset 4'][0],"--", label ='4',lw=3,color='orange')
    ax1.plot(ALE_results_ann['Subset 5'][1],ALE_results_ann['Subset 5'][0], label ='5',lw=3,color='purple')
    ax1.grid(ls='--')
    
    ax1.set_xlabel('$d_x$ (m)',fontsize=15)
    ax1.set_ylabel('(a)',labelpad=0.5,fontsize=15)
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax1.legend(fancybox=False,shadow=False)
    
    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    x_labels1 = ['','0','3.5','4.3','6.2','9.3','12.5']
    ax1.set_xticklabels(x_labels1)
    
    #dy
    ax2.plot(ALE_results_ann['Subset 2'][3],ALE_results_ann['Subset 2'][2],"-.", label ='2',lw=3,color='red')
    ax2.plot(ALE_results_ann['Subset 3'][3],ALE_results_ann['Subset 3'][2],":", label ='3',lw=3,color='green')
    ax2.plot(ALE_results_ann['Subset 4'][3],ALE_results_ann['Subset 4'][2],"--", label ='4',lw=3,color='orange')
    ax2.plot(ALE_results_ann['Subset 5'][3],ALE_results_ann['Subset 5'][2], label ='5',lw=3,color='purple')
    ax2.grid(ls='--')
    
    ax2.set_xlabel('$d_y$ (m)',fontsize=15)
    ax2.set_ylabel('(b)',labelpad=0.5,fontsize=15)
    ax2.xaxis.set_tick_params(labelsize=14)
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.legend(fancybox=False,shadow=False)
    ax2.xaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    x_labels2 = ['','0','1.8','2.2','2.5','3.3','11.4']
    ax2.set_xticklabels(x_labels2)
    
    #f
    ax3.plot(ALE_results_ann['Subset 3'][5],ALE_results_ann['Subset 3'][4],":", label ='3',lw=3,color='green')
    ax3.plot(ALE_results_ann['Subset 4'][5],ALE_results_ann['Subset 4'][4],"--", label ='4',lw=3,color='orange')
    ax3.plot(ALE_results_ann['Subset 5'][5],ALE_results_ann['Subset 5'][4], label ='5',lw=3,color='purple')
    ax3.set_xlabel('f (GHz)',fontsize=15)
    ax3.set_ylabel('(c)',labelpad=0.5,fontsize=15)
    ax3.xaxis.set_tick_params(labelsize=14)
    ax3.yaxis.set_tick_params(labelsize=14)
    ax3.legend(fancybox=False,shadow=False)
    ax3.grid(ls='--')
    
    ax3.xaxis.set_major_locator(MultipleLocator(5))
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    x_labels3 = ['','','29.0','','34.0','','40.0']
    ax3.set_xticklabels(x_labels3)
    
    #nw
    ax4.plot(ALE_results_ann['Subset 4'][7],ALE_results_ann['Subset 4'][6],"--", label ='4',lw=3,color='orange')
    ax4.plot(ALE_results_ann['Subset 5'][7],ALE_results_ann['Subset 5'][6], label ='5',lw=3,color='purple')
    
    ax4.grid(ls='--')
    ax4.set_xlabel('$n_w$',fontsize=15)
    ax4.set_ylabel('(d)',labelpad=0.5,fontsize=15)
    ax4.xaxis.set_tick_params(labelsize=14)
    ax4.yaxis.set_tick_params(labelsize=14)
    ax4.legend(fancybox=False,shadow=False)
    
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax4.xaxis.set_major_locator(MultipleLocator(0.5))
    x_labels4 = ['','1','','2','','3']
    ax4.set_xticklabels(x_labels4)
    
    #d
    ax5.plot(ALE_results_ann['Subset 5'][9],ALE_results_ann['Subset 5'][8], label ='5',lw=3,color='purple')
    ax5.grid(ls='--')
    ax5.set_xlabel('d (m)',fontsize=15)
    
    ax5.set_ylabel('(e)',labelpad=0.,fontsize=15)
    ax5.xaxis.set_tick_params(labelsize=14)
    ax5.yaxis.set_tick_params(labelsize=14)
    ax5.legend(fancybox=False,shadow=False)
    #ax5.set_xlim(0,1)
    ax5.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax5.xaxis.set_major_locator(MultipleLocator(5))
    x_labels5 = ['','2.3','4.3','5.8','7.4','9.9','12.5']
    ax5.set_xticklabels(x_labels5)
    
    #SVR
    ax6 = fig.add_subplot(spec[0,1]) # row 0 with axes spanning 2 cols on evens
    ax7 = fig.add_subplot(spec[1,1]) # row 0 with axes spanning 2 cols on evens
    ax8 = fig.add_subplot(spec[2,1]) # row 0 with axes spanning 2 cols on evens
    ax9 = fig.add_subplot(spec[3,1]) # row 0 with axes spanning 2 cols on evens
    ax10 = fig.add_subplot(spec[4,1]) # row 0 with axes spanning 2 cols on evens
    
    #nw
    ax6.plot(ALE_results_svr['Subset 2'][1],ALE_results_svr['Subset 2'][0],"+-", label ='1',lw=3,color='blue')
    ax6.plot(ALE_results_svr['Subset 2'][1],ALE_results_svr['Subset 2'][0],"-.", label ='2',lw=3,color='red')
    ax6.plot(ALE_results_svr['Subset 3'][1],ALE_results_svr['Subset 3'][0],":", label ='3',lw=3,color='green')
    ax6.plot(ALE_results_svr['Subset 4'][1],ALE_results_svr['Subset 4'][0],"--", label ='4',lw=3,color='orange')
    ax6.plot(ALE_results_svr['Subset 5'][1],ALE_results_svr['Subset 5'][0], label ='5',lw=3,color='purple')
    
    ax6.grid(ls='--')
    ax6.set_xlabel('$n_w$',fontsize=15)
    ax6.set_ylabel('(a)',labelpad=0.5,fontsize=15)
    ax6.xaxis.set_tick_params(labelsize=14)
    ax6.yaxis.set_tick_params(labelsize=14)
    ax6.legend(fancybox=False,shadow=False,loc="lower right")
    ax6.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax6.xaxis.set_major_locator(MultipleLocator(0.5))
    x_labels6 = ['','1','','2','','3']
    ax6.set_xticklabels(x_labels6)
    
    #dy
    ax7.plot(ALE_results_svr['Subset 2'][3],ALE_results_svr['Subset 2'][2],"-.", label ='2',lw=3,color='red')
    ax7.plot(ALE_results_svr['Subset 3'][3],ALE_results_svr['Subset 3'][2],":", label ='3',lw=3,color='green')
    ax7.plot(ALE_results_svr['Subset 4'][3],ALE_results_svr['Subset 4'][2],"--", label ='4',lw=3,color='orange')
    ax7.plot(ALE_results_svr['Subset 5'][3],ALE_results_svr['Subset 5'][2], label ='5',lw=3,color='purple')
    ax7.grid(ls='--')
    
    ax7.set_xlabel('$d_y$ (m)',fontsize=15)
    ax7.set_ylabel('(b)',labelpad=0.5,fontsize=15)
    ax7.xaxis.set_tick_params(labelsize=14)
    ax7.yaxis.set_tick_params(labelsize=14)
    ax7.legend(fancybox=False,shadow=False)
    
    ax7.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax7.xaxis.set_major_locator(MultipleLocator(5))
    x_labels7 = ['','0','1.8','2.2','2.5','3.3','11.4']
    ax7.set_xticklabels(x_labels7)
    
    #dx
    ax8.plot(ALE_results_svr['Subset 3'][5],ALE_results_svr['Subset 3'][4],":", label ='3',lw=3,color='green')
    ax8.plot(ALE_results_svr['Subset 4'][5],ALE_results_svr['Subset 4'][4],"--", label ='4',lw=3,color='orange')
    ax8.plot(ALE_results_svr['Subset 5'][5],ALE_results_svr['Subset 5'][4], label ='5',lw=3,color='purple')
    ax8.grid(ls='--')
    
    ax8.set_xlabel('$d_x$ (m)',fontsize=15)
    ax8.set_ylabel('(c)',labelpad=0.5,fontsize=15)
    ax8.xaxis.set_tick_params(labelsize=14)
    ax8.yaxis.set_tick_params(labelsize=14)
    ax8.legend(fancybox=False,shadow=False)
    
    ax8.xaxis.set_major_locator(MultipleLocator(5))
    ax8.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    x_labels8 = ['','0','3.5','4.3','6.2','9.3','12.5']
    ax8.set_xticklabels(x_labels8)
    
    
    #f
    ax9.plot(ALE_results_svr['Subset 4'][7],ALE_results_svr['Subset 4'][6],"--", label ='4',lw=3,color='orange')
    ax9.plot(ALE_results_svr['Subset 4'][7],ALE_results_svr['Subset 4'][6], label ='5',lw=3,color='purple')
    ax9.grid(ls='--')
    
    ax9.set_xlabel('f (GHz)',fontsize=15)
    ax9.set_ylabel('(d)',labelpad=0.5,fontsize=15)
    ax9.xaxis.set_tick_params(labelsize=14)
    ax9.yaxis.set_tick_params(labelsize=14)
    ax9.legend(fancybox=False,shadow=False)
    
    ax9.xaxis.set_major_locator(MultipleLocator(5))
    ax9.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    x_labels9 = ['','','29.0','','34.0','','40.0']
    ax9.set_xticklabels(x_labels9)
    
    #d
    ax10.plot(ALE_results_svr['Subset 5'][9],ALE_results_svr['Subset 5'][8],label ='5',lw=3,color='purple')
    ax10.grid(ls='--')
    
    ax10.set_xlabel('d (m)',fontsize=15)
    ax10.set_ylabel('(e)',labelpad=0.5,fontsize=15)
    ax10.xaxis.set_tick_params(labelsize=14)
    ax10.yaxis.set_tick_params(labelsize=14)
    ax10.legend(fancybox=False,shadow=False)
    
    ax10.xaxis.set_major_locator(MultipleLocator(5))
    ax10.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    x_labels10 = ['','2.3','4.3','5.8','7.4','9.9','12.5']
    ax10.set_xticklabels(x_labels10)
    
    #RF
    ax11 = fig.add_subplot(spec[0,2]) # row 0 with axes spanning 2 cols on evens
    ax12 = fig.add_subplot(spec[1,2]) # row 0 with axes spanning 2 cols on evens
    ax13 = fig.add_subplot(spec[2,2]) # row 0 with axes spanning 2 cols on evens
    ax14 = fig.add_subplot(spec[3,2]) # row 0 with axes spanning 2 cols on evens
    ax15 = fig.add_subplot(spec[4,2]) # row 0 with axes spanning 2 cols on evens
    
    #d
    ax11.plot(ALE_results_rf['Subset 1'][1],ALE_results_rf['Subset 1'][0],"+-", label ='1',lw=3,color='blue')
    ax11.plot(ALE_results_rf['Subset 2'][1],ALE_results_rf['Subset 2'][0],"-.", label ='2',lw=3,color='red')
    ax11.plot(ALE_results_rf['Subset 3'][1],ALE_results_rf['Subset 3'][0],":", label ='3',lw=3,color='green')
    ax11.plot(ALE_results_rf['Subset 4'][1],ALE_results_rf['Subset 4'][0],"--", label ='4',lw=3,color='orange')
    ax11.plot(ALE_results_rf['Subset 5'][1],ALE_results_rf['Subset 5'][0], label ='5',lw=3,color='purple')
    ax11.grid(ls='--')
    
    ax11.set_xlabel('d (m)',fontsize=15)
    ax11.set_ylabel('(a)',labelpad=0.5,fontsize=15)
    ax11.xaxis.set_tick_params(labelsize=14)
    ax11.yaxis.set_tick_params(labelsize=14)
    ax11.legend(fancybox=False,shadow=False)
    
    ax11.xaxis.set_major_locator(MultipleLocator(5))
    ax11.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    x_labels11 = ['','2.3','4.3','5.8','7.4','9.9','12.5']
    ax11.set_xticklabels(x_labels11)
    
    #nw
    ax12.plot(ALE_results_rf['Subset 2'][3],ALE_results_rf['Subset 2'][2],"-.", label ='2',lw=3,color='red')
    ax12.plot(ALE_results_rf['Subset 3'][3],ALE_results_rf['Subset 3'][2],":", label ='3',lw=3,color='green')
    ax12.plot(ALE_results_rf['Subset 4'][3],ALE_results_rf['Subset 4'][2],"--", label ='4',lw=3,color='orange')
    ax12.plot(ALE_results_rf['Subset 5'][3],ALE_results_rf['Subset 5'][2], label ='5',lw=3,color='purple')
    ax12.grid(ls='--')
    
    ax12.set_xlabel('$n_w$',fontsize=15)
    ax12.set_ylabel('(b)',labelpad=0.5,fontsize=15)
    ax12.xaxis.set_tick_params(labelsize=14)
    ax12.yaxis.set_tick_params(labelsize=14)
    ax12.legend(fancybox=False,shadow=False)
    
    ax12.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax12.xaxis.set_major_locator(MultipleLocator(0.5))
    x_labels12 = ['','1','','2','','3']
    ax12.set_xticklabels(x_labels12)
    
    #f
    ax13.plot(ALE_results_rf['Subset 3'][5],ALE_results_rf['Subset 3'][4],":", label ='3',lw=3,color='green')
    ax13.plot(ALE_results_rf['Subset 4'][5],ALE_results_rf['Subset 4'][4],"--", label ='4',lw=3,color='orange')
    ax13.plot(ALE_results_rf['Subset 5'][5],ALE_results_rf['Subset 5'][4], label ='5',lw=3,color='purple')
    ax13.grid(ls='--')
    
    ax13.set_xlabel('f (GHz)',fontsize=15)
    ax13.set_ylabel('(c)',labelpad=0.5,fontsize=15)
    ax13.xaxis.set_tick_params(labelsize=14)
    ax13.yaxis.set_tick_params(labelsize=14)
    ax13.legend(fancybox=False,shadow=False)
    
    ax13.xaxis.set_major_locator(MultipleLocator(5))
    ax13.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    x_labels13 = ['','','29.0','','34.0','','40.0']
    ax13.set_xticklabels(x_labels13)
    
    #dy
    ax14.plot(ALE_results_rf['Subset 4'][7],ALE_results_rf['Subset 4'][6],"--", label ='4',lw=3,color='orange')
    ax14.plot(ALE_results_rf['Subset 5'][7],ALE_results_rf['Subset 5'][6], label ='5',lw=3,color='purple')
    ax14.grid(ls='--')
    
    ax14.set_xlabel('$d_y$ (m)',fontsize=15)
    ax14.set_ylabel('(d)',labelpad=0.5,fontsize=15)
    ax14.xaxis.set_tick_params(labelsize=14)
    ax14.yaxis.set_tick_params(labelsize=14)
    ax14.legend(fancybox=False,shadow=False)
    
    ax14.xaxis.set_major_locator(MultipleLocator(5))
    ax14.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    x_labels14 = ['','0','1.3','2.2','2.5','3.3','11.4']
    ax14.set_xticklabels(x_labels14)
    
    #dx
    ax15.plot(ALE_results_rf['Subset 5'][9],ALE_results_rf['Subset 5'][8], label ='5',lw=3,color='purple')
    ax15.grid(ls='--')
    
    ax15.set_xlabel('$d_x$ (m)',fontsize=15)
    ax15.set_ylabel('(e)',labelpad=0.5,fontsize=15)
    ax15.xaxis.set_tick_params(labelsize=14)
    ax15.yaxis.set_tick_params(labelsize=14)
    ax15.legend(fancybox=False,shadow=False)
    #ax15.set_xlim(0,1)
    ax15.xaxis.set_major_locator(MultipleLocator(5))
    ax15.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    x_labels15 = ['','0','3.5','4.3','6.2','9.3','12.5']
    ax15.set_xticklabels(x_labels15)
    
    
    #GTB
    ax16 = fig.add_subplot(spec[0,3]) # row 0 with axes spanning 2 cols on evens
    ax17 = fig.add_subplot(spec[1,3]) # row 0 with axes spanning 2 cols on evens
    ax18 = fig.add_subplot(spec[2,3]) # row 0 with axes spanning 2 cols on evens
    ax19 = fig.add_subplot(spec[3,3]) # row 0 with axes spanning 2 cols on evens
    ax20 = fig.add_subplot(spec[4,3]) # row 0 with axes spanning 2 cols on evens
    
    #d
    ax16.plot(ALE_results_gtb['Subset 1'][1],ALE_results_gtb['Subset 1'][0],"+-", label ='1',lw=3,color='blue')
    ax16.plot(ALE_results_gtb['Subset 2'][1],ALE_results_gtb['Subset 2'][0],"-.", label ='2',lw=3,color='red')
    ax16.plot(ALE_results_gtb['Subset 3'][1],ALE_results_gtb['Subset 3'][0],":", label ='3',lw=3,color='green')
    ax16.plot(ALE_results_gtb['Subset 4'][1],ALE_results_gtb['Subset 4'][0],"--", label ='4',lw=3,color='orange')
    ax16.plot(ALE_results_gtb['Subset 5'][1],ALE_results_gtb['Subset 5'][0], label ='5',lw=3,color='purple')
    ax16.grid(ls='--')
    
    ax16.set_xlabel('d (m)',fontsize=15)
    ax16.set_ylabel('(a)',labelpad=0.5,fontsize=15)
    ax16.xaxis.set_tick_params(labelsize=14)
    ax16.yaxis.set_tick_params(labelsize=14)
    ax16.legend(fancybox=False,shadow=False)
    
    ax16.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax16.xaxis.set_major_locator(MultipleLocator(5))
    x_labels16 = ['','2.6','4.3','5.8','7.4','9.9','12.5']
    ax16.set_xticklabels(x_labels16)
    
    #f
    ax17.plot(ALE_results_gtb['Subset 2'][3],ALE_results_gtb['Subset 2'][2],"-.", label ='2',lw=3,color='red')
    ax17.plot(ALE_results_gtb['Subset 3'][3],ALE_results_gtb['Subset 3'][2],":", label ='3',lw=3,color='green')
    ax17.plot(ALE_results_gtb['Subset 4'][3],ALE_results_gtb['Subset 4'][2],"--", label ='4',lw=3,color='orange')
    ax17.plot(ALE_results_gtb['Subset 5'][3],ALE_results_gtb['Subset 5'][2], label ='5',lw=3,color='purple')
    ax17.grid(ls='--')
    
    ax17.set_xlabel('f (GHz)',fontsize=15)
    ax17.set_ylabel('(b)',labelpad=0.5,fontsize=15)
    ax17.xaxis.set_tick_params(labelsize=14)
    ax17.yaxis.set_tick_params(labelsize=14)
    ax17.legend(fancybox=False,shadow=False)
    
    ax17.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax17.xaxis.set_major_locator(MultipleLocator(5))
    x_labels17 = ['','','29.0','','34.0','','40.0']
    ax17.set_xticklabels(x_labels17)
    
    #nw
    ax18.plot(ALE_results_gtb['Subset 3'][5],ALE_results_gtb['Subset 3'][4],":", label ='3',lw=3,color='green')
    ax18.plot(ALE_results_gtb['Subset 4'][5],ALE_results_gtb['Subset 4'][4],"--", label ='4',lw=3,color='orange')
    ax18.plot(ALE_results_gtb['Subset 5'][5],ALE_results_gtb['Subset 5'][4], label ='5',lw=3,color='purple')
    ax18.grid(ls='--')
    
    ax18.set_xlabel('$n_w$',fontsize=15)
    ax18.set_ylabel('(c)',labelpad=0.5,fontsize=15)
    ax18.xaxis.set_tick_params(labelsize=14)
    ax18.yaxis.set_tick_params(labelsize=14)
    ax18.legend(fancybox=False,shadow=False)
    
    ax18.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax18.xaxis.set_major_locator(MultipleLocator(0.5))
    x_labels18 = ['','1','','2','','3']
    ax18.set_xticklabels(x_labels18)
    
    #dy
    ax19.plot(ALE_results_gtb['Subset 4'][7],ALE_results_gtb['Subset 4'][6],"--", label ='4',lw=3,color='orange')
    ax19.plot(ALE_results_gtb['Subset 5'][7],ALE_results_gtb['Subset 5'][6], label ='5',lw=3,color='purple')
    ax19.grid(ls='--') 
    
    ax19.set_xlabel('$d_y$ (m)',labelpad=0.5,fontsize=15)
    ax19.set_ylabel('(d)',fontsize=15)
    ax19.xaxis.set_tick_params(labelsize=14)
    ax19.yaxis.set_tick_params(labelsize=14)
    ax19.legend(fancybox=False,shadow=False)
    
    ax19.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax19.xaxis.set_major_locator(MultipleLocator(5))
    x_labels19 = ['','0','1.8','2.2','2.5','3.3','11.4']
    ax19.set_xticklabels(x_labels19)
    
    #dx
    ax20.plot(ALE_results_gtb['Subset 5'][9],ALE_results_gtb['Subset 5'][8], label ='5',lw=3,color='purple')
    ax20.grid(ls='--')
    
    ax20.set_xlabel('$d_x$ (m)',fontsize=15)
    ax20.set_ylabel('(e)',labelpad=0.5,fontsize=15)
    ax20.legend(fancybox=False,shadow=False)
    ax20.xaxis.set_tick_params(labelsize=14)
    ax20.yaxis.set_tick_params(labelsize=14)
    
    ax20.xaxis.set_major_locator(MultipleLocator(5))
    ax20.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    x_labels20 = ['','0','3.5','4.3','6.2','9.3','12.5']
    ax20.set_xticklabels(x_labels20)
    
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.36,
                        hspace=0.6
                        )
    
    #fig.supylabel('Centered Accumulated Local Effect',fontsize=18,fontweight=20,x=0.037)
    
    plt.figtext(0.18, 0.91, '(1) ANN', ha='center', va='center',fontsize='15',weight='bold')
    plt.figtext(0.39,  0.91, '(2) SVR', ha='center', va='center',fontsize='15',weight='bold')
    plt.figtext(0.61,  0.91, '(3) RF', ha='center', va='center',fontsize='15',weight='bold')
    plt.figtext(0.82, 0.91, '(4) GTB', ha='center', va='center',fontsize='15',weight='bold')
    
    plt.savefig('ALE_2.eps',format='eps',dpi=1200)
    plt.show()
    plt.close()
    
    
    #------------------------------------------------------------------------------------------------------
    """
    Results of the ML models for Table 9. Performances and complexities of the four final regression having optimized
    predictors subsets and hyperparameters.
    """
    
    #ANN
    print('-------------------------------------------------------------')
    #performance and IML indicators
    
    data = [
    ["Training set",final_model_ann["train"][0],final_model_ann["train"][1],final_model_ann["train"][2],final_model_ann["train"][3]],
    ["Testing set",final_model_ann["test"][0],final_model_ann["test"][1],final_model_ann["test"][2],final_model_ann["test"][3]],
    ["IML (IAS - MEC)",final_model_ann["IML"][0],final_model_ann["IML"][1]]
    ]
    print(tabulate(data, headers=[" Subset of 4 predictors (ANN)","RMSE [dB]","MAPE [%]","SD [dB]","R2"])) 
    
    #SVR
    print('-------------------------------------------------------------')
    #performance and IML indicators
    
    data = [
    ["Training set",final_model_svr["train"][0],final_model_svr["train"][1],final_model_svr["train"][2],final_model_svr["train"][3]],
    ["Testing set",final_model_svr["test"][0],final_model_svr["test"][1],final_model_svr["test"][2],final_model_svr["test"][3]],
    ["IML (IAS - MEC)",final_model_svr["IML"][0],final_model_svr["IML"][1]]
    ]
    print(tabulate(data, headers=[" Subset of 5 predictors (SVR)","RMSE [dB]","MAPE [%]","SD [dB]","R2"])) 
    
    
    #RF
    print('-------------------------------------------------------------')
    #performance and IML indicators
    
    data = [
    ["Training set",final_model_rf["train"][0],final_model_rf["train"][1],final_model_rf["train"][2],final_model_rf["train"][3]],
    ["Testing set",final_model_rf["test"][0],final_model_rf["test"][1],final_model_rf["test"][2],final_model_rf["test"][3]],
    ["IML (IAS - MEC)",final_model_rf["IML"][0],final_model_rf["IML"][1]]
    ]
    print(tabulate(data, headers=[" Subset of 4 predictors (RF)","RMSE [dB]","MAPE [%]","SD [dB]","R2"])) 
    
    #GTB
    print('-------------------------------------------------------------')
    #performance and IML indicators
    
    data = [
    ["Training set",final_model_gtb["train"][0],final_model_gtb["train"][1],final_model_gtb["train"][2],final_model_gtb["train"][3]],
    ["Testing set",final_model_gtb["test"][0],final_model_gtb["test"][1],final_model_gtb["test"][2],final_model_gtb["test"][3]],
    ["IML (IAS - MEC)",final_model_gtb["IML"][0],final_model_gtb["IML"][1]]
    ]
    print(tabulate(data, headers=[" Subset of 3 predictors (GTB)","RMSE [dB]","MAPE [%]","SD [dB]","R2"])) 

    
    #------------------------------------------------------------------------------------------------------
    print('---------------Figure 6 --------------------')
    fig = plt.figure(figsize=(9,11))
    spec = gridspec.GridSpec(ncols=2, nrows=3) # 6 columns evenly divides both 2 & 3
    
    ax1 = fig.add_subplot(spec[0,0]) # row 0 with axes spanning 2 cols on evens
    ax2 = fig.add_subplot(spec[0,1]) # row 0 with axes spanning 2 cols on evens
    ax3 = fig.add_subplot(spec[1,0]) # row 0 with axes spanning 2 cols on evens
    ax4 = fig.add_subplot(spec[1,1]) # row 0 with axes spanning 2 cols on evens
    ax5 = fig.add_subplot(spec[2,0]) # row 0 with axes spanning 2 cols on evens
    ax6 = fig.add_subplot(spec[2,1]) # row 0 with axes spanning 2 cols on evens
    
    ax1.scatter(matrix_27GHz['d'],matrix_27GHz['y_target'],s=90,marker='o',color='blue',facecolor='none',linewidths=1.5,label='Measured')
    ax1.scatter(matrix_27GHz['d'],matrix_27GHz['y_pred'],s=90,marker='*',color='green',label='Predicted')
    ax2.scatter(matrix_27GHz['nw'],matrix_27GHz['y_target'],s=90,marker='o',color='blue',facecolor='none',linewidths=1.5,label='Measured')
    ax2.scatter(matrix_27GHz['nw'],matrix_27GHz['y_pred'],s=90,marker='*',color='green',label='Predicted')
    
    ax3.scatter(matrix_33GHz['d'],matrix_33GHz['y_target'],s=90,marker='o',color='red',facecolor='none',linewidths=1.5,label='Measured')
    ax3.scatter(matrix_33GHz['d'],matrix_33GHz['y_pred'],s=90,marker='*',color='purple',label='Predicted')
    ax4.scatter(matrix_33GHz['nw'],matrix_33GHz['y_target'],s=90,marker='o',color='red',facecolor='none',linewidths=1.5,label='Measured')
    ax4.scatter(matrix_33GHz['nw'],matrix_33GHz['y_pred'],s=90,marker='*',color='purple',label='Predicted')
    
    ax5.scatter(matrix_40GHz['d'],matrix_40GHz['y_target'],s=90,marker='o',color='chocolate',facecolor='none',linewidths=1.5,label='Measured')
    ax5.scatter(matrix_40GHz['d'],matrix_40GHz['y_pred'],s=90,marker='*',color='darkslategray',label='Predicted')
    ax6.scatter(matrix_40GHz['nw'],matrix_40GHz['y_target'],s=90,marker='o',color='chocolate',facecolor='none',linewidths=1.5,label='Measured')
    ax6.scatter(matrix_40GHz['nw'],matrix_40GHz['y_pred'],s=90,marker='*',color='darkslategray',label='Predicted')
    
    ax1.xaxis.label.set_size(20)
    
    ax1.tick_params(axis='both', labelsize=13)
    ax2.tick_params(axis='both', labelsize=13)
    ax3.tick_params(axis='both', labelsize=13)
    ax4.tick_params(axis='both', labelsize=13)
    ax5.tick_params(axis='both', labelsize=13)
    ax6.tick_params(axis='both', labelsize=13)
    
    ax2.xaxis.set_major_locator(MultipleLocator(0.5))
    x_labels2 = ['','1','','2','','3']
    ax2.set_xticklabels(x_labels2)
    
    ax4.xaxis.set_major_locator(MultipleLocator(0.5))
    x_labels4 = ['','1','','2','','3']
    ax4.set_xticklabels(x_labels4)
    
    ax6.xaxis.set_major_locator(MultipleLocator(0.5))
    x_labels6 = ['','1','','2','','3']
    ax6.set_xticklabels(x_labels6)
    
    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')
    ax3.legend(loc='lower right')
    
    plt.subplots_adjust(right=1.2)
    ax1.grid(linestyle='--')
    ax2.grid(linestyle='--')
    ax3.grid(linestyle='--')
    ax4.grid(linestyle='--')
    ax5.grid(linestyle='--')
    ax6.grid(linestyle='--')
    
    ax1.set_xlim([2,12])
    ax3.set_xlim([2,12])
    ax5.set_xlim([2,12])
    
    ax1.set_ylabel('Path-loss [dB]',fontsize=15)
    ax1.set_xlabel('$d$ [m]',fontsize=15)
    ax2.set_ylabel('Path-loss [dB]',fontsize=15)
    ax2.set_xlabel('$n_w$',fontsize=15)
    ax3.set_ylabel('Path-loss [dB]',fontsize=15)
    ax3.set_xlabel('$d$ [m]',fontsize=15)
    ax4.set_ylabel('Path-loss [dB]',fontsize=15)
    ax4.set_xlabel('$n_w$',fontsize=15)
    ax5.set_ylabel('Path-loss [dB]',fontsize=15)
    ax5.set_xlabel('$d$ [m]',fontsize=15)
    ax6.set_ylabel('Path-loss [dB]',fontsize=15)
    ax6.set_xlabel('$n_w$',fontsize=15)
    
    ax1.legend(fancybox=False,shadow=False,loc='upper left')
    ax2.legend(fancybox=False,shadow=False)
    ax3.legend(fancybox=False,shadow=False)
    ax4.legend(fancybox=False,shadow=False)
    ax5.legend(fancybox=False,shadow=False)
    ax6.legend(fancybox=False,shadow=False)
    
    plt.figtext(0.65, 0.91, '(a) 27 GHz', ha='center', va='center',fontsize='13',weight='bold')
    plt.figtext(0.65, 0.63, '(b) 33 GHz', ha='center', va='center',fontsize='13',weight='bold')
    plt.figtext(0.65, 0.35, '(b) 40 GHz', ha='center', va='center',fontsize='13',weight='bold')
    
    plt.subplots_adjust(
                        wspace=0.25,
                        hspace=0.4
                        )
    
    plt.savefig('pl_dist_nw.eps',format='eps',dpi=1200,bbox_inches='tight')
    
    plt.show()
    plt.close()
    
 
    #------------------------------------------------------------------------------------------------------
    """
    Results for the Table 10: Performance of the proposed GTB-based PL model when the models are training
    considering a set of transmitters and the test employs a different one.
    """
    
    print('---------------Table 10--------------------')
    data = [
    ["Test 1",results_generalization_gtb["Tx 1"][0],results_generalization_gtb["Tx 1"][1]],
    ["Test 2",results_generalization_gtb["Tx 2"][0],results_generalization_gtb["Tx 2"][1]],
    ["Test 3",results_generalization_gtb["Tx 3"][0],results_generalization_gtb["Tx 3"][1]],
    ["Test 4",results_generalization_gtb["Tx 4"][0],results_generalization_gtb["Tx 4"][1]],
    ["Test 5",results_generalization_gtb["Tx 5"][0],results_generalization_gtb["Tx 5"][1]],
    ["Test 6",results_generalization_gtb["Tx 6"][0],results_generalization_gtb["Tx 6"][1]]
    ]
    print(tabulate(data, headers=["","RMSE Training [dB]","RMSE Testing [dB]"])) 
    
 
    #------------------------------------------------------------------------------------------------------
    """
    Results for the Table 11: Generalization capability analysis for the GTB-based PL model. The models
    are trained using two transmitters and evaluated (tested) on the four remaining ones.
    """
    
    print('---------------Table 11--------------------')
    data = [
    ["Test 1",results_generalization_gtb_tx["Tx 1"][0],results_generalization_gtb_tx["Tx 1"][1]],
    ["Test 2",results_generalization_gtb_tx["Tx 2"][0],results_generalization_gtb_tx["Tx 2"][1]],
    ["Test 3",results_generalization_gtb_tx["Tx 3"][0],results_generalization_gtb_tx["Tx 3"][1]],
    ["Test 4",results_generalization_gtb_tx["Tx 4"][0],results_generalization_gtb_tx["Tx 4"][1]]
    ]
    print(tabulate(data, headers=["","RMSE Training [dB]","RMSE Testing [dB]"])) 
    
    
    #------------------------------------------------------------------------------------------------------
    """
    Results for the Table 12: GTB and empirical PL models performance using the database from Yonsei
    University, Korea.
    """
    
    print('---------------Table 12--------------------')
    #performance and IML indicators
    
    data = [
    ["Floor 2 (Training set)",results_yonsei["train_floor2 - test_floor3_4"][0]],
    ["Floor 3 (Testing set)",results_yonsei["train_floor2 - test_floor3_4"][1]],
    ["Floor 4 (Testing set)",results_yonsei["train_floor2 - test_floor3_4"][2]],
    ["Floor 3 (Training set)",results_yonsei["train_floor3 - test_floor2_4"][0]],
    ["Floor 2 (Testing set)",results_yonsei["train_floor3 - test_floor2_4"][1]],
    ["Floor 4 (Testing set)",results_yonsei["train_floor3 - test_floor2_4"][2]],
    ["Floor 4 (Training set)",results_yonsei["train_floor4 - test_floor2_3"][0]],
    ["Floor 2 (Testing set)",results_yonsei["train_floor4 - test_floor2_3"][1]],
    ["Floor 3 (Testing set)",results_yonsei["train_floor4 - test_floor2_3"][2]],
    ]
    print(tabulate(data, headers=["Floor","RMSE [dB]",])) 
    
    return print("Finished Section 5")