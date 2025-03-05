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

path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\CODE GITHUB - THESIS\V2I\R2-ML-models.csv"
df = pd.read_csv(path)

fig = plt.figure(figsize=(14,11))
spec = gridspec.GridSpec(ncols=2, nrows=2) # 6 columns evenly divides both 2 & 3

ax1 = fig.add_subplot(spec[0,0])
ax2 = fig.add_subplot(spec[0,1]) 
ax3 = fig.add_subplot(spec[1,0]) 
ax4 = fig.add_subplot(spec[1,1]) 


p1 =110
p2 = 0
ax1.plot([p1, p2], [p1, p2], '--',linewidth=1.5,c='navy')
ax2.plot([p1, p2], [p1, p2], '--',linewidth=1.5,c='navy')
ax3.plot([p1, p2], [p1, p2], '--',linewidth=1.5,c='navy')
ax4.plot([p1, p2], [p1, p2], '--',linewidth=1.5,c='navy')

ax1.scatter(df['pl_target'],df['pl_pred_ann'],s=200, label= 'ANN: $R^2 = -0.51$',c='olivedrab',marker='+')
ax2.scatter(df['pl_target'],df['pl_pred_svr'],s=130, label= 'SVR: $R^2 = -1.33$',c='darksalmon',marker='D',linewidths=0)
ax3.scatter(df['pl_target'],df['pl_pred_rf'],s=180, label= 'RF: $R^2 = 0.58$',c='turquoise')
ax4.scatter(df['pl_target'],df['pl_pred_gtb'],s=180,label= 'GTB: $R^2 = 0.69$',c='crimson',marker='*')

#ax1.set_xlabel('Measured PL [dB]',fontsize=27)
#ax1.set_ylabel('Predicted PL [dB]',fontsize=27)

#ax2.set_xlabel('Measured PL [dB]',fontsize=27)
#ax2.set_ylabel('Predicted PL [dB]',fontsize=27)

#ax3.set_xlabel('Measured PL [dB]',fontsize=27)
#ax3.set_ylabel('Predicted PL [dB]',fontsize=27)

#ax4.set_xlabel('Measured PL [dB]',fontsize=27)
#ax4.set_ylabel('Predicted PL [dB]',fontsize=27)

ax1.set_xlim(0,110)
ax1.set_ylim(0,110)
ax1.grid(ls='--')
ax1.legend(fontsize=26)

ax2.set_ylim(0,110)
ax2.set_xlim(0,110)
ax2.grid(ls='--')
ax2.legend(fontsize=26)

ax3.set_ylim(0,110)
ax3.set_xlim(0,110)
ax3.grid(ls='--')
ax3.legend(fontsize=26)

ax4.set_ylim(0,110)
ax4.set_xlim(0,110)
ax4.grid(ls='--')
ax4.legend(loc='lower right',fontsize=26)

ax1.tick_params(axis='x', labelsize=33)
ax1.tick_params(axis='y', labelsize=33)
ax2.tick_params(axis='x', labelsize=33)
ax2.tick_params(axis='y', labelsize=33)
ax3.tick_params(axis='x', labelsize=33)
ax3.tick_params(axis='y', labelsize=33)
ax4.tick_params(axis='x', labelsize=33)
ax4.tick_params(axis='y', labelsize=33)

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.25
                    )

fig.supylabel('Predicted PL [dB]',fontsize=34,fontweight=20,x=0.0)
fig.supxlabel('Measured PL [dB]',fontsize=34,fontweight=20,y=0.0)

#fig.tight_layout()
plt.savefig('r2_comparison_V2I_new.eps',format='eps',dpi=1200)
plt.show()
plt.close()

#%%

path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\CODE GITHUB - THESIS\V2I\R2-ML-models.csv"
df = pd.read_csv(path)

fig = plt.figure(figsize=(14,11))
spec = gridspec.GridSpec(ncols=2, nrows=2) # 6 columns evenly divides both 2 & 3

ax1 = fig.add_subplot(spec[0,0])
ax2 = fig.add_subplot(spec[0,1]) 
ax3 = fig.add_subplot(spec[1,0]) 
ax4 = fig.add_subplot(spec[1,1]) 


p1 = 130#max(max(df["pl_target"]), max(df["pl_target"]))
p2 = 0#min(min(df["pl_target"]), min(df["pl_target"]))
ax1.plot([p1, p2], [p1, p2], '--',linewidth=1.5,c='navy')
ax2.plot([p1, p2], [p1, p2], '--',linewidth=1.5,c='navy')
ax3.plot([p1, p2], [p1, p2], '--',linewidth=1.5,c='navy')
ax4.plot([p1, p2], [p1, p2], '--',linewidth=1.5,c='navy')

ax1.scatter(df['pl_target'],df['pl_pred_ann'],s=90, label= '$R^2:-0.51$',c='olivedrab',marker='+')
ax2.scatter(df['pl_target'],df['pl_pred_svr'],s=60, label= '$R^2:-1.33$',c='darksalmon',marker='D',linewidths=0)
ax3.scatter(df['pl_target'],df['pl_pred_rf'],s=100, label= '$R^2:0.58$',c='turquoise')
ax4.scatter(df['pl_target'],df['pl_pred_gtb'],s=120,label= '$R^2:0.69$',c='crimson',marker='*')

ax1.set_xlabel('Measured PL [dB]',fontsize=21)
ax1.set_ylabel('Predicted PL [dB]',fontsize=21)
ax1.set_title('(a) ANN', fontsize=21)

ax2.set_xlabel('Measured PL [dB]',fontsize=21)
ax2.set_ylabel('Predicted PL [dB]',fontsize=21)
ax2.set_title('(b) SVR', fontsize=21)

ax3.set_xlabel('Measured PL [dB]',fontsize=21)
ax3.set_ylabel('Predicted PL [dB]',fontsize=21)
ax3.set_title('(c) RF', fontsize=23)

ax4.set_xlabel('Measured PL [dB]',fontsize=21)
ax4.set_ylabel('Predicted PL [dB]',fontsize=21)
ax4.set_title('(d) GTB', fontsize=21)

ax1.set_ylim(0,130)
ax1.set_xlim(0,130)
ax1.grid(ls='--')
ax1.legend(fontsize=20)

ax2.set_ylim(0,130)
ax2.set_xlim(0,130)
ax2.grid(ls='--')
ax2.legend(fontsize=20)

ax3.set_ylim(0,130)
ax3.set_xlim(0,130)
ax3.grid(ls='--')
ax3.legend(fontsize=20)

ax4.set_xlim(0,130)
ax4.set_ylim(0,130)
ax4.grid(ls='--')
ax4.legend(fontsize=20)

ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)
ax2.tick_params(axis='x', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)
ax3.tick_params(axis='x', labelsize=20)
ax3.tick_params(axis='y', labelsize=20)
ax4.tick_params(axis='x', labelsize=20)
ax4.tick_params(axis='y', labelsize=20)

fig.tight_layout()

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.4
                    )


plt.savefig('r2_comparison_V2I.eps',format='eps',dpi=1200)
plt.show()
plt.close()

#%%

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec


fig = plt.figure(figsize=(20,6))
spec = gridspec.GridSpec(ncols=3, nrows=1) # 6 columns evenly divides both 2 & 3

ax1 = fig.add_subplot(spec[0,0]) # row 0 with axes spanning 2 cols on evens
ax2 = fig.add_subplot(spec[0,1]) # row 0 with axes spanning 2 cols on evens
ax3 = fig.add_subplot(spec[0,2]) # row 0 with axes spanning 2 cols on evens

path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\R2_plot_gen_outdoor\R2_final_model_V2I.csv"
df = pd.read_csv(path)
df.head()

pl_target_750 = df["pl_target_750"]
pl_pred_gtb_750 = df["pl_pred_gtb_750"]
pl_pred_log_750 = df["pl_pred_log_750"]

pl_target_2500 = df["pl_target_2500"]
pl_pred_gtb_2500 = df["pl_pred_gtb_2500"]
pl_pred_log_2500 = df["pl_pred_log_2500"]

pl_target_3500 = df["pl_target_3500"]
pl_pred_gtb_3500 = df["pl_pred_gtb_3500"]
pl_pred_log_3500 = df["pl_pred_log_3500"]

#750 MHz
p1 = 100
p2 = 15
ax1.plot([p1, p2], [p1, p2], '--',linewidth=1,c='navy')
ax1.scatter(pl_target_750, pl_pred_log_750,marker='+', label="Log-distance",color='maroon',s=130)
ax1.scatter(pl_target_750, pl_pred_gtb_750,marker='*', label="GTB",color='coral',s=130)

ax1.grid(ls='--')
ax1.legend(fontsize=23)

ax1.set_xlabel('Measured PL [dB]',size=26)
ax1.set_ylabel('Predicted PL [dB]',size=26)
ax1.set_title('(a) 735 MHz',size=26)

#2500 MHz

p1 = 100
p2 = 55
ax2.plot([p1, p2], [p1, p2], '--',linewidth=1,c='navy')
ax2.scatter(pl_target_2500, pl_pred_log_2500,marker='+', label="Log-distance",color='maroon',s=130)
ax2.scatter(pl_target_2500, pl_pred_gtb_2500,marker='*', label="GTB",color='coral',s=130)

ax2.grid(ls='--')
ax2.legend(fontsize=23)

ax2.set_xlabel('Measured PL [dB]',size=26)
ax2.set_ylabel('Predicted PL [dB]',size=26)
ax2.set_title('(b) 2.54 GHz',size=26)


p1 = 100
p2 = 55
#ax1.plot([p1, p2], [p1, p2], '--',linewidth=1,label= 'Best fit',c='navy')
#ax2.plot([p1, p2], [p1, p2], '--',linewidth=1,label= 'Best fit',c='navy')
ax3.plot([p1, p2], [p1, p2], '--',linewidth=1,c='navy')
ax3.scatter(pl_target_3500, pl_pred_log_3500,marker='+', label="Log-distance",color='maroon',s=130)
ax3.scatter(pl_target_3500, pl_pred_gtb_3500,marker='*', label="GTB",color='coral',s=130)
#ax3.scatter(pl_gtb_750, pl_pred_gtb_750,marker='x',label="GTB, RMSE: 9.90 dB",color='teal',s=25)

ax3.grid(ls='--')
ax3.legend(fontsize=23)

ax3.set_xlabel('Measured PL [dB]',size=26)
ax3.set_ylabel('Predicted PL [dB]',size=26)
ax3.set_title('(c) 3.5 GHz',size=26)

ax1.tick_params(axis='x', labelsize=26)
ax1.tick_params(axis='y', labelsize=26)
ax2.tick_params(axis='x', labelsize=26)
ax2.tick_params(axis='y', labelsize=26)
ax3.tick_params(axis='x', labelsize=26)
ax3.tick_params(axis='y', labelsize=26)

ax1.set_xlim([15,100])
ax1.set_ylim([15,100])
ax2.set_xlim([55,100])
ax2.set_ylim([55,100])
ax3.set_xlim([55,100])
ax3.set_ylim([55,100])

fig.tight_layout()
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.01,
                    hspace=0.5
                    )


fig.tight_layout()
plt.savefig('R2_V2I_frequency.eps',format='eps',dpi=600)
plt.show()
plt.close()

#%%

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(7,5))

RMSE=[10.87,1.63,2.12,0.91,0.42,0.31,1.26,0.11,0.19,0.17,0.14,0.2]
IAS=[0,0,-0.13,-0.03,0.03,0.02,-0.27,0.22,0.04,-0.06,0.01,0.12]
MEC=[1,-0.23,-1.16,0.55,0.3,-0.35,0.04,-0.98,0.57,0.07,-0.1,-0.31]

x=[1,2,3,4,5,6,7,8,9,10,11,12]


plt.plot(x,RMSE,"-o",label='$\Delta RMSE_p [dB]$',linewidth=3,markersize=8)
#plt.plot(x,IAS,"-o",label='$\Delta IAS_p$',linewidth=2.1,markersize=5)
#plt.plot(x,MEC,"-o",label='$\Delta\overline{MEC}_p$',linewidth=2.1,markersize=5)

plt.xlim([0,13])
plt.ylim([-1,12])

x=[1,2,3,4,5,6,7,8,9,10,11,12]

plt.xticks(x)

plt.xlabel('Predictor subsets',fontsize=22)
plt.ylabel('Marginal contribution',fontsize=22)
plt.grid(ls='--')
plt.legend(fontsize="21")
plt.tick_params(axis='x', labelsize=21)
plt.tick_params(axis='y', labelsize=21)

fig.tight_layout()
plt.savefig('MC_V2I.eps',format='eps',dpi=600)
plt.show()
plt.close()

