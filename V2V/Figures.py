# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 17:15:50 2023

@author: Yoiz Nuñez
"""
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import matplotlib.gridspec as gridspec

path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\R2_plot_gen_outdoor\R2_final_model_V2V.csv"
df = pd.read_csv(path)

fig = plt.figure(figsize=(15,11))
spec = gridspec.GridSpec(ncols=1, nrows=1) # 6 columns evenly divides both 2 & 3

ax1 = fig.add_subplot(spec[0,0])

p1 = 85
p2 = 0
ax1.plot([p1, p2], [p1, p2], '--',linewidth=1.5,label= 'Best fit',c='navy')

ax1.scatter(df['y_test'],df['y_pred_log'],s=160, label= 'Log-distance, RMSE: 9.74 dB, $R^2: -0.77$',c='olivedrab',marker='+')
ax1.scatter(df['y_test'],df['y_pred_ann'],s=140, label= 'ANN,RMSE: 16.66 dB, $R^2:-4.17$ (2 predictors)',c='darksalmon',marker='D',linewidths=0)
ax1.scatter(df['y_test'],df['y_pred_svr'],s=180, label= 'SVR, RMSE: 8.50 dB, $R^2: -0.35$ (4 predictors)',c='turquoise')
ax1.scatter(df['y_test'],df['y_pred_gtb'],s=180,label= 'GTB, RMSE: 7.78 dB, $R^2: -0.13$ (4 predictors)',c='crimson',marker='x')
ax1.scatter(df['y_test'],df['y_pred_rf'],s=180,label= 'RF, RMSE: 7.53 dB, $R^2: -0.06$ (4 predictors)',c='blue',marker='.')
ax1.scatter(df['y_test'],df['y_pred_cnn'],s=180,label= 'pre-trained CNN, RMSE: 6.97 dB, $R^2 = 0.10$',c='crimson',marker='*')

ax1.set_xlabel('Measured PL [dB]',fontsize=33)
ax1.set_ylabel('Predicted PL [dB]',fontsize=33)
#ax1.set_title('(a) ANN', fontsize=20)

ax1.set_ylim(0,85)
ax1.set_xlim(0,85)
ax1.grid(ls='--')
ax1.legend(fontsize=19)

ax1.tick_params(axis='x', labelsize=33)
ax1.tick_params(axis='y', labelsize=33)

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.35,
                    hspace=0.5
                    )

fig.tight_layout()
plt.savefig('r2_comparison_V2V.eps',format='eps',dpi=1200)
plt.show()
plt.close()

#%%

path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\R2_plot_gen_outdoor\R2_final_model_V2V.csv"
df = pd.read_csv(path)



fig = plt.figure(figsize=(14,13))


spec = gridspec.GridSpec(ncols=1, nrows=1) # 6 columns evenly divides both 2 & 3

ax1 = fig.add_subplot(spec[0,0])

p1 = 90
p2 = 0
ax1.plot([p1, p2], [p1, p2], '--',linewidth=2,label= 'Best fit',c='navy')

ax1.scatter(df['y_test'],df['y_pred_log'],s=450, label= 'Log-distance, RMSE: 9.74 dB, $R^2:-0.77$',c='olivedrab',marker='+')
#ax1.scatter(df['y_test'],df['y_pred_ann'],s=140, label= 'ANN,RMSE: 16.66 dB, $R^2:-4.17$ (2 predictors)',c='darksalmon',marker='D',linewidths=0)
#ax1.scatter(df['y_test'],df['y_pred_svr'],s=180, label= 'SVR, RMSE: 8.50 dB, $R^2: -0.35$ (4 predictors)',c='turquoise')
#ax1.scatter(df['y_test'],df['y_pred_gtb'],s=180,label= 'GTB, RMSE: 7.78 dB, $R^2: -0.13$ (4 predictors)',c='crimson',marker='x')
ax1.scatter(df['y_test'],df['y_pred_rf'],s=450,label= 'RF, RMSE: 7.53 dB, $R^2:-0.06$ (4 predictors)',c='blue',marker='.')
ax1.scatter(df['y_test'],df['y_pred_cnn'],s=400,label= 'pre-trained CNN, RMSE: 6.97 dB, $R^2:0.10$',c='crimson',marker='*')


ax1.set_xlabel('Measured PL [dB]',fontsize=39)
ax1.set_ylabel('Predicted PL [dB]',fontsize=39)
#ax1.set_title('(a) ANN', fontsize=20)


ax1.set_ylim(40,90)
ax1.set_xlim(40,90)
ax1.grid(ls='--')
ax1.legend(fontsize=26)

ax1.tick_params(axis='x', labelsize=39)
ax1.tick_params(axis='y', labelsize=39)


# Add a legend
#pos = ax1.get_position()v 
#ax1.set_position([pos.x0, pos.y0, pos.width, pos.height *1])
#ax1.legend(fontsize=23,
#    loc='upper center', 
#    bbox_to_anchor=(0.5,1.9),
#    ncol=1, 
#)


#fig.tight_layout()
plt.savefig('r2_comparison_V2V_final.eps',format='eps',dpi=1200)
plt.show()
plt.close()