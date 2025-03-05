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

path=r"C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\CODE GITHUB - THESIS\Macrocell sub-6GHz\Section 5.4.2\R2 - ML_models.csv"
df = pd.read_csv(path)


fig = plt.figure(figsize=(14,11))
spec = gridspec.GridSpec(ncols=2, nrows=2) # 6 columns evenly divides both 2 & 3

ax1 = fig.add_subplot(spec[0,0])
ax2 = fig.add_subplot(spec[0,1]) 
ax3 = fig.add_subplot(spec[1,0]) 
ax4 = fig.add_subplot(spec[1,1]) 


p1 = 185
p2 = 50
ax1.plot([p1, p2], [p1, p2], '--',linewidth=1.5,c='navy')
ax2.plot([p1, p2], [p1, p2], '--',linewidth=1.5,c='navy')
ax3.plot([p1, p2], [p1, p2], '--',linewidth=1.5,c='navy')
ax4.plot([p1, p2], [p1, p2], '--',linewidth=1.5,c='navy')

ax1.scatter(df['pl_target'],df['pl_pred_ann'],s=90, label= '$R^2: 0.10$',c='olivedrab',marker='+')
ax2.scatter(df['pl_target'],df['pl_pred_svr'],s=60, label= '$R^2: 0.05$',c='darksalmon',marker='D',linewidths=0)
ax3.scatter(df['pl_target'],df['pl_pred_rf'],s=100, label= '$R^2: 0.31$',c='turquoise')
ax4.scatter(df['pl_target'],df['pl_pred_gtb'],s=120,label= '$R^2: 0.56$',c='crimson',marker='*')

ax1.set_xlabel('Measured path loss [dB]',fontsize=20)
ax1.set_ylabel('Predicted path loss [dB]',fontsize=20)
ax1.set_title('(a) ANN', fontsize=20)

ax2.set_xlabel('Measured path loss [dB]',fontsize=20)
ax2.set_ylabel('Predicted path loss [dB]',fontsize=20)
ax2.set_title('(b) SVR', fontsize=20)

ax3.set_xlabel('Measured path loss [dB]',fontsize=20)
ax3.set_ylabel('Predicted path loss [dB]',fontsize=20)
ax3.set_title('(c) RF', fontsize=20)

ax4.set_xlabel('Measured path loss [dB]',fontsize=20)
ax4.set_ylabel('Predicted path loss [dB]',fontsize=20)
ax4.set_title('(d) GTB', fontsize=20)

ax1.set_ylim(50,185)
ax1.set_xlim(50,185)
ax1.grid(ls='--')
ax1.legend(fontsize=18)

ax2.set_ylim(50,185)
ax2.set_xlim(50,185)
ax2.grid(ls='--')
ax2.legend(fontsize=18)

ax3.set_ylim(50,185)
ax3.set_xlim(50,185)
ax3.grid(ls='--')
ax3.legend(fontsize=18)

ax4.set_ylim(50,185)
ax4.set_xlim(50,185)
ax4.grid(ls='--')
ax4.legend(fontsize=18)

ax1.tick_params(axis='x', labelsize=19)
ax1.tick_params(axis='y', labelsize=19)
ax2.tick_params(axis='x', labelsize=19)
ax2.tick_params(axis='y', labelsize=19)
ax3.tick_params(axis='x', labelsize=19)
ax3.tick_params(axis='y', labelsize=19)
ax4.tick_params(axis='x', labelsize=19)
ax4.tick_params(axis='y', labelsize=19)

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.35,
                    hspace=0.5
                    )


#fig.tight_layout()
plt.savefig('r2_comparison_macrocell.eps',format='eps',dpi=1200)
plt.show()
plt.close()