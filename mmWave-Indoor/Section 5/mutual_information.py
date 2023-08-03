# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
from sklearn.model_selection import train_test_split

#!pip install matplotlib --upgrade
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MultipleLocator


def mutual_information_plot(df):

    def mi(X_train,y_train,x_var,y_var):
      
      X=X_train
      X=X.squeeze()
      y=y_train
      y=y.squeeze()
      
      #if predictors is a continuous value, then bin is 20, if predictor is categoric, then bin is 3
      if x_var =="numerical":
         bin_x=20
      else: 
         bin_x=3
    
      if y_var =="numerical":
         bin_y=20
      else: 
         bin_y=3
    
      bins=[bin_x,bin_y]
    
      p_xy, xedges, yedges = np.histogram2d(X, y, bins=bins, density=True)
      # p(x)p(y)Calculation
      p_x, _ = np.histogram(X, bins=xedges, density=True)
      p_y, _ = np.histogram(y, bins=yedges, density=True)
      p_x_y = p_x[:, np.newaxis] * p_y
      #dx and dy
      dx = xedges[1] - xedges[0]
      dy = yedges[1] - yedges[0]
      #Integral element
      elem = p_xy * np.ma.log(p_xy / p_x_y)
      #Mutual information and p(x, y), p(x)p(y)Output
      mi_value= np.sum(elem * dx * dy)
      return mi_value
    
    
    fig = plt.figure(figsize=(17,11))
    spec = gridspec.GridSpec(ncols=5, nrows=3) # 6 columns evenly divides both 2 & 3
    
    ax1 = fig.add_subplot(spec[0,0]) # row 0 with axes spanning 2 cols on evens
    ax2 = fig.add_subplot(spec[0,1]) # row 0 with axes spanning 2 cols on evens
    ax3 = fig.add_subplot(spec[0,2]) # row 0 with axes spanning 2 cols on evens
    ax4 = fig.add_subplot(spec[0,3]) # row 0 with axes spanning 2 cols on evens
    ax5 = fig.add_subplot(spec[0,4]) # row 0 with axes spanning 2 cols on evens
    
    ax1.scatter(df['f'],df['PL'],s=18, marker='o')
    ax1.grid(ls='--')
    ax1.set_xlabel('f',fontsize=16)
    ax1.set_ylabel('PL',fontsize=16)
    ax1.xaxis.set_tick_params(labelsize=16)
    ax1.yaxis.set_tick_params(labelsize=16)
    ax1.plot((0, 1), (0, 1), transform=ax1.transAxes, ls='--',c='k',lw='0.7')
    
    blanks = '\n'.join(['x'*10])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.57, 0.06, blanks, color='none', transform=ax1.transAxes, bbox=props)
    
    # overlay statistics with titles left-aligned and numbers right-aligned
    # MI: Select the predictors to be assessed
    X = df['f']
    y = df['PL'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mi_value= mi(X_train,y_train,"numerical","numerical")
    stats_txt = 'MI:'+ str(round(mi_value, 2))
    
    props = dict(boxstyle='square', facecolor='none', edgecolor='none')
    ax1.text(0.57, 0.05, stats_txt, transform=ax1.transAxes, bbox=props, ha='left',fontsize=15)
    
    ax2.scatter(df['d'],df['PL'],s=18, marker='o')
    ax2.grid(ls='--')
    ax2.set_xlabel('d',fontsize=16)
    ax2.set_ylabel('PL',fontsize=16)
    ax2.xaxis.set_tick_params(labelsize=16)
    ax2.yaxis.set_tick_params(labelsize=16)
    ax2.plot((0, 1), (0, 1), transform=ax2.transAxes, ls='--',c='k',lw='0.7')
    blanks = '\n'.join(['x'*10])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.57, 0.05, blanks, color='none', transform=ax2.transAxes, bbox=props)
    # overlay statistics with titles left-aligned and numbers right-aligned
    
    X = df['d'] 
    y = df['PL']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mi_value= mi(X_train,y_train,"numerical","numerical")
    stats_txt = 'MI:'+ str(round(mi_value, 2))
    
    props = dict(boxstyle='square', facecolor='none', edgecolor='none')
    ax2.text(0.57, 0.05, stats_txt, transform=ax2.transAxes, bbox=props, ha='left',fontsize=15)
    
    ax3.scatter(df['dx'],df['PL'],s=18, marker='o')
    ax3.grid(ls='--')
    ax3.set_xlabel('$d_x$',fontsize=16)
    ax3.set_ylabel('PL',fontsize=16)
    ax3.xaxis.set_tick_params(labelsize=15)
    ax3.yaxis.set_tick_params(labelsize=15)
    ax3.plot((0, 1), (0, 1), transform=ax3.transAxes, ls='--',c='k',lw='0.7')
    blanks = '\n'.join(['x'*10])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax3.text(0.57, 0.05, blanks, color='none', transform=ax3.transAxes, bbox=props)
    # overlay statistics with titles left-aligned and numbers right-aligned
    
    X = df['dx'] 
    y = df['PL'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mi_value= mi(X_train,y_train,"numerical","numerical")
    stats_txt = 'MI:'+ str(round(mi_value, 2))
    
    props = dict(boxstyle='square', facecolor='none', edgecolor='none')
    ax3.text(0.57, 0.05, stats_txt, transform=ax3.transAxes, bbox=props, ha='left',fontsize=15)
    
    ax4.scatter(df['dy'],df['PL'],s=18, marker='o')
    ax4.grid(ls='--')
    ax4.set_xlabel('$d_y$',fontsize=16)
    ax4.set_ylabel('PL',fontsize=16)
    ax4.xaxis.set_tick_params(labelsize=16)
    ax4.yaxis.set_tick_params(labelsize=16)
    ax4.plot((0, 1), (0, 1), transform=ax4.transAxes, ls='--',c='k',lw='0.7')
    blanks = '\n'.join(['x'*10])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax4.text(0.57, 0.05, blanks, color='none', transform=ax4.transAxes, bbox=props)
    # overlay statistics with titles left-aligned and numbers right-aligned
    
    X = df['dy'] 
    y = df['PL'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mi_value= mi(X_train,y_train,"numerical","numerical")
    stats_txt = 'MI:'+ str(round(mi_value, 2))
    
    props = dict(boxstyle='square', facecolor='none', edgecolor='none')
    ax4.text(0.57, 0.05, stats_txt, transform=ax4.transAxes, bbox=props, ha='left',fontsize=15)
    
    ax5.scatter(df['nw'],df['PL'],s=18, marker='o')
    ax5.grid(ls='--')
    ax5.set_xlabel('$n_w$',fontsize=16)
    ax5.set_ylabel('PL',fontsize=16)
    ax5.xaxis.set_tick_params(labelsize=16)
    ax5.yaxis.set_tick_params(labelsize=16)
    ax5.plot((0, 1), (0, 1), transform=ax5.transAxes, ls='--',c='k',lw='0.7')
    blanks = '\n'.join(['x'*10])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax5.text(0.57, 0.05, blanks, color='none', transform=ax5.transAxes, bbox=props)
    # overlay statistics with titles left-aligned and numbers right-aligned
    X = df['nw'] 
    y = df['PL']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mi_value= mi(X_train,y_train,"categorical","numerical")
    stats_txt = 'MI:'+ str(round(mi_value, 2))
    
    props = dict(boxstyle='square', facecolor='none', edgecolor='none')
    ax5.text(0.57, 0.05, stats_txt, transform=ax5.transAxes, bbox=props, ha='left',fontsize=15)
    
    ax6 = fig.add_subplot(spec[1,0]) # row 0 with axes spanning 2 cols on evens
    ax7 = fig.add_subplot(spec[1,1]) # row 0 with axes spanning 2 cols on evens
    ax8 = fig.add_subplot(spec[1,2]) # row 0 with axes spanning 2 cols on evens
    ax9 = fig.add_subplot(spec[1,3]) # row 0 with axes spanning 2 cols on evens
    ax10 = fig.add_subplot(spec[1,4]) # row 0 with axes spanning 2 cols on evens
    
    ax6.scatter(df['dx'],df['d'],s=18, marker='o')
    ax6.grid(ls='--')
    ax6.set_xlabel('$d_x$',fontsize=16)
    ax6.set_ylabel('$d$',fontsize=16)
    ax6.xaxis.set_tick_params(labelsize=16)
    ax6.yaxis.set_tick_params(labelsize=16)
    ax6.plot((0, 1), (0, 1), transform=ax6.transAxes, ls='--',c='k',lw='0.7')
    blanks = '\n'.join(['x'*10])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax6.text(0.57, 0.05, blanks, color='none', transform=ax6.transAxes, bbox=props)
    # overlay statistics with titles left-aligned and numbers right-aligned
    X = df['dx']
    y = df['d'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mi_value= mi(X_train,y_train,"numerical","numerical")
    stats_txt = 'MI:'+ str(round(mi_value, 2))
    
    props = dict(boxstyle='square', facecolor='none', edgecolor='none')
    ax6.text(0.57, 0.05, stats_txt, transform=ax6.transAxes, bbox=props, ha='left',fontsize=15)
    ax6.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
    ax7.scatter(df['dy'],df['d'],s=18, marker='o')
    ax7.grid(ls='--')
    ax7.set_xlabel('$d_y$',fontsize=16)
    ax7.set_ylabel('$d$',fontsize=16,linespacing=0,labelpad=1)
    ax7.xaxis.set_tick_params(labelsize=16)
    ax7.yaxis.set_tick_params(labelsize=16)
    ax7.plot((0, 1), (0, 1), transform=ax7.transAxes, ls='--',c='k',lw='0.7')
    blanks = '\n'.join(['x'*10])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax7.text(0.57, 0.05, blanks, color='none', transform=ax7.transAxes, bbox=props)
    # overlay statistics with titles left-aligned and numbers right-aligned
    X = df['dy']
    y = df['d'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mi_value= mi(X_train,y_train,"numerical","numerical")
    stats_txt = 'MI:'+ str(round(mi_value, 2))
    
    props = dict(boxstyle='square', facecolor='none', edgecolor='none')
    ax7.text(0.57, 0.05, stats_txt, transform=ax7.transAxes, bbox=props, ha='left',fontsize=15)
    ax7.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
    ax8.scatter(df['dy'],df['dx'],s=18, marker='o')
    ax8.grid(ls='--')
    ax8.set_xlabel('$d_y$',fontsize=16)
    ax8.set_ylabel('$d_x$',fontsize=16)
    ax8.xaxis.set_tick_params(labelsize=16)
    ax8.yaxis.set_tick_params(labelsize=16)
    ax8.plot((0, 1), (0, 1), transform=ax8.transAxes, ls='--',c='k',lw='0.7')
    blanks = '\n'.join(['x'*10])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax8.text(0.57, 0.05, blanks, color='none', transform=ax8.transAxes, bbox=props)
    # overlay statistics with titles left-aligned and numbers right-aligned
    X = df['dy']
    y = df['dx'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mi_value= mi(X_train,y_train,"numerical","numerical")
    stats_txt = 'MI:'+ str(round(mi_value, 2))
    
    props = dict(boxstyle='square', facecolor='none', edgecolor='none')
    ax8.text(0.57, 0.05, stats_txt, transform=ax8.transAxes, bbox=props, ha='left',fontsize=15)
    
    ax9.scatter(df['nw'],df['d'],s=18, marker='o')
    ax9.grid(ls='--')
    ax9.set_xlabel('$n_w$',fontsize=16)
    ax9.set_ylabel('$d$',fontsize=16)
    ax9.xaxis.set_tick_params(labelsize=16)
    ax9.yaxis.set_tick_params(labelsize=16)
    ax9.plot((0, 1), (0, 1), transform=ax9.transAxes, ls='--',c='k',lw='0.7')
    blanks = '\n'.join(['x'*10])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax9.text(0.57, 0.05, blanks, color='none', transform=ax9.transAxes, bbox=props)
    # overlay statistics with titles left-aligned and numbers right-aligned
    X = df['nw']
    y = df['d'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mi_value= mi(X_train,y_train,"categorical","numerical")
    stats_txt = 'MI:'+ str(round(mi_value, 2))
    
    props = dict(boxstyle='square', facecolor='none', edgecolor='none')
    ax9.text(0.57, 0.05, stats_txt, transform=ax9.transAxes, bbox=props, ha='left',fontsize=15)
    ax9.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    
    ax10.scatter(df['nw'],df['dx'],s=18, marker='o')
    ax10.grid(ls='--')
    ax10.set_xlabel('$n_w$',fontsize=16)
    ax10.set_ylabel('$d_x$',fontsize=16)
    ax10.xaxis.set_tick_params(labelsize=16)
    ax10.yaxis.set_tick_params(labelsize=16)
    ax10.plot((0, 1), (0, 1), transform=ax10.transAxes, ls='--',c='k',lw='0.7')
    blanks = '\n'.join(['x'*10])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax10.text(0.57, 0.05, blanks, color='none', transform=ax10.transAxes, bbox=props)
    # overlay statistics with titles left-aligned and numbers right-aligned
    X = df['dx']
    y = df['nw'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mi_value= mi(X_train,y_train,"numerical","categorical")
    stats_txt = 'MI:'+ str(round(mi_value, 2))
    
    props = dict(boxstyle='square', facecolor='none', edgecolor='none')
    ax10.text(0.57, 0.05, stats_txt, transform=ax10.transAxes, bbox=props, ha='left',fontsize=15)
    
    ax11 = fig.add_subplot(spec[2,0]) # row 0 with axes spanning 2 cols on evens
    ax12 = fig.add_subplot(spec[2,1]) # row 0 with axes spanning 2 cols on evens
    ax13 = fig.add_subplot(spec[2,2]) # row 0 with axes spanning 2 cols on evens
    ax14 = fig.add_subplot(spec[2,3]) # row 0 with axes spanning 2 cols on evens
    ax15 = fig.add_subplot(spec[2,4]) # row 0 with axes spanning 2 cols on evens
    
    ax11.scatter(df['nw'],df['dy'],s=18, marker='o')
    ax11.grid(ls='--')
    ax11.set_xlabel('$n_w$',fontsize=16)
    ax11.set_ylabel('$d_y$',fontsize=16)
    ax11.xaxis.set_tick_params(labelsize=16)
    ax11.yaxis.set_tick_params(labelsize=16)
    ax11.plot((0, 1), (0, 1), transform=ax11.transAxes, ls='--',c='k',lw='0.7')
    blanks = '\n'.join(['x'*10])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax11.text(0.57, 0.05, blanks, color='none', transform=ax11.transAxes, bbox=props)
    # overlay statistics with titles left-aligned and numbers right-aligned
    X = df['nw']
    y = df['dy'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mi_value= mi(X_train,y_train,"categorical","numerical")
    stats_txt = 'MI:'+ str(round(mi_value, 2))
    
    props = dict(boxstyle='square', facecolor='none', edgecolor='none')
    ax11.text(0.57, 0.05, stats_txt, transform=ax11.transAxes, bbox=props, ha='left',fontsize=15)
    
    ax12.scatter(df['d'],df['f'],s=18, marker='o')
    ax12.grid(ls='--')
    ax12.set_xlabel('$d$',fontsize=16)
    ax12.set_ylabel('$f$',fontsize=16)
    ax12.xaxis.set_tick_params(labelsize=16)
    ax12.yaxis.set_tick_params(labelsize=16)
    ax12.plot((0, 1), (0, 1), transform=ax12.transAxes, ls='--',c='k',lw='0.7')
    blanks = '\n'.join(['x'*10])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax12.text(0.57, 0.05, blanks, color='none', transform=ax12.transAxes, bbox=props)
    # overlay statistics with titles left-aligned and numbers right-aligned
    X = df['d']
    y = df['f'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mi_value= mi(X_train,y_train,"numerical","numerical")
    stats_txt = 'MI:'+ str(round(mi_value, 2))
    
    props = dict(boxstyle='square', facecolor='none', edgecolor='none')
    ax12.text(0.57, 0.05, stats_txt, transform=ax12.transAxes, bbox=props, ha='left',fontsize=15)
    
    
    ax13.scatter(df['dx'],df['f'],s=18, marker='o')
    ax13.grid(ls='--')
    ax13.set_xlabel('$d_x$',fontsize=16)
    ax13.set_ylabel('$f$',fontsize=16)
    ax13.xaxis.set_tick_params(labelsize=16)
    ax13.yaxis.set_tick_params(labelsize=16)
    ax13.plot((0, 1), (0, 1), transform=ax13.transAxes, ls='--',c='k',lw='0.7')
    blanks = '\n'.join(['x'*10])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax13.text(0.57, 0.05, blanks, color='none', transform=ax13.transAxes, bbox=props)
    # overlay statistics with titles left-aligned and numbers right-aligned
    X = df['dx']
    y = df['f'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mi_value= mi(X_train,y_train,"numerical","numerical")
    stats_txt = 'MI:'+ str(round(mi_value, 2))
    
    props = dict(boxstyle='square', facecolor='none', edgecolor='none')
    ax13.text(0.57, 0.05, stats_txt, transform=ax13.transAxes, bbox=props, ha='left',fontsize=15)
    
    ax14.scatter(df['dy'],df['f'],s=18, marker='o')
    ax14.grid(ls='--')
    ax14.set_xlabel('$d_y$',fontsize=16)
    ax14.set_ylabel('$f$',fontsize=16)
    ax14.xaxis.set_tick_params(labelsize=16)
    ax14.yaxis.set_tick_params(labelsize=16)
    ax14.plot((0, 1), (0, 1), transform=ax14.transAxes, ls='--',c='k',lw='0.7')
    blanks = '\n'.join(['x'*10])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax14.text(0.57, 0.05, blanks, color='none', transform=ax14.transAxes, bbox=props)
    # overlay statistics with titles left-aligned and numbers right-aligned
    X = df['dy']
    y = df['f'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mi_value= mi(X_train,y_train,"numerical","numerical")
    stats_txt = 'MI:'+ str(round(mi_value, 2))
    
    props = dict(boxstyle='square', facecolor='none', edgecolor='none')
    ax14.text(0.57, 0.05, stats_txt, transform=ax14.transAxes, bbox=props, ha='left',fontsize=15)
    
    ax15.scatter(df['nw'],df['f'],s=18, marker='o')
    ax15.grid(ls='--')
    ax15.set_xlabel('$n_w$',fontsize=16)
    ax15.set_ylabel('$f$',fontsize=16)
    ax15.xaxis.set_tick_params(labelsize=16)
    ax15.yaxis.set_tick_params(labelsize=16)
    ax15.plot((0, 1), (0, 1), transform=ax15.transAxes, ls='--',c='k',lw='0.7')
    blanks = '\n'.join(['x'*10])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax15.text(0.57, 0.05, blanks, color='none', transform=ax15.transAxes, bbox=props)
    
    X = df['nw']
    y = df['f'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mi_value= mi(X_train,y_train,"categorical","numerical")
    stats_txt = 'MI:'+ str(round(mi_value, 2))
    
    props = dict(boxstyle='square', facecolor='none', edgecolor='none')
    ax15.text(0.57, 0.05, stats_txt, transform=ax15.transAxes, bbox=props, ha='left',fontsize=15)
    
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.42,
                        hspace=0.27
                        )
    
    plt.savefig('MI.eps',format='eps',dpi=1200)
    plt.show()
    plt.close()
    
    return print("Figure 3 have been plotted")

