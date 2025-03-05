# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:41:52 2023

@author: Yoiz Nuñez
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
"""
Read Data
"""

path=r"Rx2.csv"
df2 = pd.read_csv(path)
df2.head()

#RX2
x = df2.iloc[:, [0]] #distance
f = df2.iloc[:, [2]] #heigth of Tx
g = df2.iloc[:, [1]]

plt.plot(x, f, '-',linewidth=2)
plt.plot(x, g, '-',color='green',label='Vegetation profile')
plt.plot(0, 50, 'o',color='red')
plt.plot(61.79281919, 1, 'o',color='orange')

#p1
plt.plot(50.71, 9.06, '*',color='black')
#p2
plt.plot(51.64, 8.30, '*',color='black')
#p3
plt.plot(58.71, 3.02, '*',color='black')
#p4
plt.plot(62.41, 1.0, '*',color='black')

plt.grid(ls='--')
plt.xlim(0, 62.5)
plt.xlabel('Distance [m]')
plt.ylabel('Vegetation heigth [m]')
plt.legend()

#%%
path=r"Rx4.csv"
df4 = pd.read_csv(path)
df4.head()

#RX4
x = df4.iloc[:, [0]] #distance
f = df4.iloc[:, [2]] #heigth of Tx
g = df4.iloc[:, [1]]

plt.plot(x, f, '-',linewidth=2)
plt.plot(x, g, '-',color='green',label='Vegetation profile')
plt.plot(0, 50, 'o',color='red')
plt.plot(63.76073372, 1, 'o',color='orange')

#p1
plt.plot(59.96, 4, '*',color='black')
#p2
plt.plot(63.76, 1, '*',color='black')
#p3
#plt.plot(58.71, 3.02, '*',color='black')
#p4
#plt.plot(62.41, 1.0, '*',color='black')

plt.grid(ls='--')
plt.xlim(0,63.76073372)
plt.xlabel('Distance [m]')
plt.ylabel('Vegetation heigth [m]')
plt.legend()

#%%

path=r"Rx15.csv"
df15 = pd.read_csv(path)
df15.head()

#RX4
x = df15.iloc[:, [0]] #distance
f = df15.iloc[:, [2]] #heigth of Tx
g = df15.iloc[:, [1]]

plt.plot(x, f, '-',linewidth=2)
plt.plot(x, g, '-',color='green',label='Vegetation profile')
plt.plot(0, 50, 'o',color='red')
plt.plot(142.775902, 4, 'o',color='orange')

#p1
plt.plot(135.53, 6.34, '*',color='black')
#p2
plt.plot(139.28, 5.13, '*',color='black')
#p3
#plt.plot(58.71, 3.02, '*',color='black')
#p4
#plt.plot(62.41, 1.0, '*',color='black')

plt.grid(ls='--')
plt.xlim(0,142.775902)
plt.xlabel('Distance [m]')
plt.ylabel('Vegetation heigth [m]')
plt.legend()

#%%

path=r"Rx16.csv"
df16 = pd.read_csv(path)
df16.head()

#RX4
x = df16.iloc[:, [0]] #distance
f = df16.iloc[:, [2]] #heigth of Tx
g = df16.iloc[:, [1]]

plt.plot(x, f, '-',linewidth=2)
plt.plot(x, g, '-',color='green',label='Vegetation profile')
plt.plot(0, 50, 'o',color='red')
plt.plot(113.8211893, 1, 'o',color='orange')

#p1
plt.plot(109.67, 2.79, '*',color='black')
#p2
plt.plot(113.82,1, '*',color='black')
#p3
#plt.plot(58.71, 3.02, '*',color='black')
#p4
#plt.plot(62.41, 1.0, '*',color='black')

plt.grid(ls='--')
plt.xlim(0,113.8211893)
plt.xlabel('Distance [m]')
plt.ylabel('Vegetation heigth [m]')
plt.legend()

#%%

path=r"Rx1.csv"
df2 = pd.read_csv(path)
df2.head()

#RX4
x = df1.iloc[:, [0]] #distance
f = df1.iloc[:, [2]] #heigth of Tx
g = df1.iloc[:, [1]]

plt.plot(x, f, '-',linewidth=2)
plt.plot(x, g, '-',color='green',label='Vegetation profile')
plt.plot(0, 50, 'o',color='red')
plt.plot(73.85645649, 1, 'o',color='orange')

#p1
plt.plot(55.51, 13, '*',color='black')
#p2
plt.plot(55.87,12.76, '*',color='black')
#p3
plt.plot(67.74, 5.0, '*',color='black')
#p4
plt.plot(70.34, 3.30, '*',color='black')
#p5
plt.plot(72.33, 2, '*',color='black')
#p6
plt.plot(73.86, 1, '*',color='black')

plt.grid(ls='--')
plt.xlim(0,73.85645649)
plt.xlabel('Distance [m]')
plt.ylabel('Vegetation heigth [m]')
plt.legend()

#%%

path=r"Rx10.csv"
df2 = pd.read_csv(path)
df2.head()

#RX4
x = df10.iloc[:, [0]] #distance
f = df10.iloc[:, [2]] #heigth of Tx
g = df10.iloc[:, [1]]

plt.plot(x, f, '-',linewidth=2)
plt.plot(x, g, '-',color='green',label='Vegetation profile')
plt.plot(0, 50, 'o',color='red')
plt.plot(18.88753632, 1, 'o',color='orange')

#p1
plt.plot(18.11,3, '*',color='black')
#p2
plt.plot(18.89,1, '*',color='black')
#p3
#plt.plot(67.74, 5.0, '*',color='black')
#p4
#plt.plot(70.34, 3.30, '*',color='black')
#p5
#plt.plot(72.33, 2, '*',color='black')
#p6
#plt.plot(73.86, 1, '*',color='black')

plt.grid(ls='--')
plt.xlim(0,18.88753632)
plt.xlabel('Distance [m]')
plt.ylabel('Vegetation heigth [m]')
plt.legend()


#%%
idx = np.argwhere(np.diff(np.sign(f - g))).flatten()

#%%
plt.plot(d[idx], h[idx], 'ro')
plt.show()

#%%

from shapely.geometry import LineString

first_line = LineString(np.column_stack((x, f)))
second_line = LineString(np.column_stack((x, g)))
intersection = first_line.intersection(second_line)

#%%

if intersection.geom_type == 'MultiPoint':
    plt.plot(*LineString(intersection).xy, 'o')
elif intersection.geom_type == 'Point':
    plt.plot(*intersection.xy, 'o')
    

x, y = LineString(intersection).xy

#%%
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(11,6))
spec = gridspec.GridSpec(ncols=3, nrows=2) 

ax1 = fig.add_subplot(spec[0,0]) 
ax2 = fig.add_subplot(spec[0,1]) 
ax3 = fig.add_subplot(spec[0,2]) 
ax4 = fig.add_subplot(spec[1,0]) 
ax5 = fig.add_subplot(spec[1,1]) 
ax6 = fig.add_subplot(spec[1,2]) 


#RX1
path=r"Rx1.csv"
df1 = pd.read_csv(path)
df1.head()

x = df1.iloc[:, [0]] #distance
f = df1.iloc[:, [2]] #heigth of Tx
g = df1.iloc[:, [1]]

ax1.plot(x, f, '--',linewidth=2)
ax1.plot(x, g, '-',color='green',linewidth=3)
ax1.plot(0, 50, 'o',color='red',markersize=10,markeredgecolor='red',markerfacecolor='white')
ax1.plot(73.85645649, 1, 'o',color='orange',markersize=10,markeredgecolor='orange',markerfacecolor='white')

#p1
ax1.plot(55.51, 13, '*',color='black')
#p2
ax1.plot(55.87,12.76, '*',color='black')
#p3
ax1.plot(67.74, 5.0, '*',color='black')
#p4
ax1.plot(70.34, 3.30, '*',color='black')
#p5
ax1.plot(72.33, 2, '*',color='black')
#p6
ax1.plot(73.86, 1, '*',color='black')

ax1.grid(ls='--')
ax1.set_xlim(0,73.85645649)
ax1.set_xlabel('d [m]',fontsize=14)
ax1.set_ylabel('Vegetation heigth [m]',fontsize=14)
#ax1.legend()
ax1.set_title('(a) Rx1', fontsize=14)


blanks = '\n'.join(['x'*12])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax1.text(0.05, 0.3, blanks, color='none', transform=ax1.transAxes, bbox=props)
# overlay statistics with titles left-aligned and numbers right-aligned
stats_txt = '$v_{depth}$: 5.37 m'
#stats = '\n'.join([
   # '{}'.format(0.5)
#])
props = dict(boxstyle='square', facecolor='none', edgecolor='none')
ax1.text(0.05, 0.3, stats_txt, transform=ax1.transAxes, bbox=props, ha='left')


#RX2
path=r"Rx2.csv"
df2 = pd.read_csv(path)
df2.head()

#RX2
x = df2.iloc[:, [0]] #distance
f = df2.iloc[:, [2]] #heigth of Tx
g = df2.iloc[:, [1]]

ax2.plot(x, f, '--',linewidth=2)
ax2.plot(x, g, '-',color='green',linewidth=3)
ax2.plot(0, 50, 'o',color='red',markersize=10,markeredgecolor='red',markerfacecolor='white')
ax2.plot(61.79281919, 1, 'o',color='orange',markersize=10,markeredgecolor='orange',markerfacecolor='white')


#p
ax2.plot(50.71, 9.06, '*',color='black')
#p2
ax2.plot(51.64, 8.30, '*',color='black')
#p3
ax2.plot(58.71, 3.02, '*',color='black')
#p4
ax2.plot(62.41, 1.0, '*',color='black')

ax2.set_xlim(0, 62.5)
ax2.set_xlabel('d [m]',fontsize=14)
ax2.set_ylabel('Vegetation heigth [m]',fontsize=14)
ax2.grid(ls='--')
#ax2.legend()
ax2.set_title('(b) Rx2', fontsize=14)


blanks = '\n'.join(['x'*12])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax2.text(0.05, 0.3, blanks, color='none', transform=ax2.transAxes, bbox=props)
# overlay statistics with titles left-aligned and numbers right-aligned
stats_txt = '$v_{depth}$: 5.42 m'
#stats = '\n'.join([
   # '{}'.format(0.5)
#])
props = dict(boxstyle='square', facecolor='none', edgecolor='none')
ax2.text(0.05, 0.3, stats_txt, transform=ax2.transAxes, bbox=props, ha='left')

#RX4
path=r"Rx4.csv"
df4 = pd.read_csv(path)
df4.head()

#RX4
x = df4.iloc[:, [0]] #distance
f = df4.iloc[:, [2]] #heigth of Tx
g = df4.iloc[:, [1]]

ax3.plot(x, f, '--',linewidth=2)
ax3.plot(x, g, '-',color='green',linewidth=3)
ax3.plot(0, 50, 'o',color='red',markersize=10,markeredgecolor='red',markerfacecolor='white')
ax3.plot(63.76073372, 1, 'o',color='orange',markersize=10,markeredgecolor='orange',markerfacecolor='white')


#p1
ax3.plot(59.96, 4, '*',color='black')
#p2
ax3.plot(63.76, 1, '*',color='black')
#p3
#plt.plot(58.71, 3.02, '*',color='black')
#p4
#plt.plot(62.41, 1.0, '*',color='black')

ax3.grid(ls='--')
ax3.set_xlim(0,63.76073372)
ax3.set_xlabel('d [m]',fontsize=14)
ax3.set_ylabel('Vegetation heigth [m]',fontsize=14)
#ax3.legend()
ax3.set_title('(c) Rx4', fontsize=14)

blanks = '\n'.join(['x'*12])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax3.text(0.05, 0.3, blanks, color='none', transform=ax3.transAxes, bbox=props)
# overlay statistics with titles left-aligned and numbers right-aligned
stats_txt = '$v_{depth}$: 4.84 m'
#stats = '\n'.join([
   # '{}'.format(0.5)
#])
props = dict(boxstyle='square', facecolor='none', edgecolor='none')
ax3.text(0.05, 0.3, stats_txt, transform=ax3.transAxes, bbox=props, ha='left')

#RX10
path=r"Rx10.csv"
df2 = pd.read_csv(path)
df2.head()

#RX4
x = df2.iloc[:, [0]] #distance
f = df2.iloc[:, [2]] #heigth of Tx
g = df2.iloc[:, [1]]

ax4.plot(x, f, '--',linewidth=2)
ax4.plot(x, g, '-',color='green',linewidth=3)
ax4.plot(0, 50, 'o',color='red',markersize=10,markeredgecolor='red',markerfacecolor='white')
ax4.plot(18.88753632, 1, 'o',color='orange',markersize=10,markeredgecolor='orange',markerfacecolor='white')


#p1
ax4.plot(18.11,3, '*',color='black')
#p2
ax4.plot(18.89,1, '*',color='black')
#p3
#plt.plot(67.74, 5.0, '*',color='black')
#p4
#plt.plot(70.34, 3.30, '*',color='black')
#p5
#plt.plot(72.33, 2, '*',color='black')
#p6
#plt.plot(73.86, 1, '*',color='black')

ax4.grid(ls='--')
ax4.set_xlim(0,18.88753632)
ax4.set_xlabel('d [m]',fontsize=14)
ax4.set_ylabel('Vegetation heigth [m]',fontsize=14)
#ax4.legend()
ax4.set_title('(d) Rx10', fontsize=14)


blanks = '\n'.join(['x'*12])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax4.text(0.05, 0.3, blanks, color='none', transform=ax4.transAxes, bbox=props)
# overlay statistics with titles left-aligned and numbers right-aligned
stats_txt = '$v_{depth}$: 2.15 m'
#stats = '\n'.join([
   # '{}'.format(0.5)
#])
props = dict(boxstyle='square', facecolor='none', edgecolor='none')
ax4.text(0.05, 0.3, stats_txt, transform=ax4.transAxes, bbox=props, ha='left')

#RX15
path=r"Rx15.csv"
df15 = pd.read_csv(path)
df15.head()

#RX4
x = df15.iloc[:, [0]] #distance
f = df15.iloc[:, [2]] #heigth of Tx
g = df15.iloc[:, [1]]

ax5.plot(x, f, '--',linewidth=2)
ax5.plot(x, g, '-',color='green',linewidth=3)
ax5.plot(0, 50, 'o',color='red',markersize=10,markeredgecolor='red',markerfacecolor='white')
ax5.plot(142.775902, 4, 'o',color='orange',markersize=10,markeredgecolor='orange',markerfacecolor='white')


#p1
ax5.plot(135.53, 6.34, '*',color='black')
#p2
ax5.plot(139.28, 5.13, '*',color='black')
#p3
#plt.plot(58.71, 3.02, '*',color='black')
#p4
#plt.plot(62.41, 1.0, '*',color='black')

ax5.grid(ls='--')
ax5.set_xlim(0,142.775902)
ax5.set_xlabel('d [m]',fontsize=14)
ax5.set_ylabel('Vegetation heigth [m]',fontsize=14)
#ax5.legend()
ax5.set_title('(e) Rx15', fontsize=14)


blanks = '\n'.join(['x'*12])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax5.text(0.05, 0.3, blanks, color='none', transform=ax5.transAxes, bbox=props)
# overlay statistics with titles left-aligned and numbers right-aligned
stats_txt = '$v_{depth}$: 4.16 m'
#stats = '\n'.join([
   # '{}'.format(0.5)
#])
props = dict(boxstyle='square', facecolor='none', edgecolor='none')
ax5.text(0.05, 0.3, stats_txt, transform=ax5.transAxes, bbox=props, ha='left')


path=r"Rx16.csv"
df16 = pd.read_csv(path)
df16.head()

#RX4
x = df16.iloc[:, [0]] #distance
f = df16.iloc[:, [2]] #heigth of Tx
g = df16.iloc[:, [1]]

ax6.plot(x, f, '--',linewidth=2)
ax6.plot(x, g, '-',color='green',linewidth=3)
ax6.plot(0, 50, 'o',color='red',markersize=10,markeredgecolor='red',markerfacecolor='white')
ax6.plot(113.8211893, 1, 'o',color='orange',markersize=10,markeredgecolor='orange',markerfacecolor='white')


#p1
ax6.plot(109.67, 2.79, '*',color='black')
#p2
ax6.plot(113.82,1, '*',color='black')
#p3
#plt.plot(58.71, 3.02, '*',color='black')
#p4
#plt.plot(62.41, 1.0, '*',color='black')

ax6.grid(ls='--')
ax6.set_xlim(0,113.8211893)
ax6.set_xlabel('d [m]',fontsize=14)
ax6.set_ylabel('Vegetation heigth [m]',fontsize=14)
#ax6.legend()
ax6.set_title('(f) Rx16', fontsize=14)

blanks = '\n'.join(['x'*12])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax6.text(0.05, 0.3, blanks, color='none', transform=ax6.transAxes, bbox=props)
# overlay statistics with titles left-aligned and numbers right-aligned
stats_txt = '$v_{depth}$: 4.52 m'
#stats = '\n'.join([
   # '{}'.format(0.5)
#])
props = dict(boxstyle='square', facecolor='none', edgecolor='none')
ax6.text(0.05, 0.3, stats_txt, transform=ax6.transAxes, bbox=props, ha='left')


ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
ax3.tick_params(axis='x', labelsize=14)
ax3.tick_params(axis='y', labelsize=14)
ax4.tick_params(axis='x', labelsize=14)
ax4.tick_params(axis='y', labelsize=14)
ax5.tick_params(axis='x', labelsize=14)
ax5.tick_params(axis='y', labelsize=14)
ax6.tick_params(axis='x', labelsize=14)
ax6.tick_params(axis='y', labelsize=14)


plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.5
                    )

labels = ['Direct path Tx-Rx', 'Vegetation profile', 'Tx','Rx','Point of vegetation intersection']
fig.legend(labels, loc='center', bbox_to_anchor=(0.5,0.97), ncol=len(labels), bbox_transform=fig.transFigure, fontsize=13)
plt.savefig('vd.eps',format='eps',dpi=1200)
plt.show()
plt.close()

#%%

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(8,6))
spec = gridspec.GridSpec(ncols=2, nrows=2) 

ax1 = fig.add_subplot(spec[0,0]) 
ax2 = fig.add_subplot(spec[0,1]) 
ax3 = fig.add_subplot(spec[1,0]) 
ax4 = fig.add_subplot(spec[1,1]) 

path=r"Outdoor_data_vegetationdepth_qgis_final.csv"
df = pd.read_csv(path)
df.head()

f = df.iloc[:, [0]] 
d = df.iloc[:, [1]] 
h = df.iloc[:, [2]] 
vd = df.iloc[:, [5]] 
pl=df.iloc[:, [3]] 

ax1.scatter(f,pl,s=20)
ax1.set_xlabel('f [GHz]',fontsize=12)
ax1.set_ylabel('Measured PL [dB]',fontsize=12)
ax1.plot((0, 1), (0, 1), transform=ax1.transAxes, ls='--',c='k',lw='0.7')

blanks = '\n'.join(['x'*7])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax1.text(0.76, 0.05, blanks, color='none', transform=ax1.transAxes, bbox=props)
# overlay statistics with titles left-aligned and numbers right-aligned
stats_txt = 'MI:0.40'
#stats = '\n'.join([
   # '{}'.format(0.5)
#])
props = dict(boxstyle='square', facecolor='none', edgecolor='none')
ax1.text(0.76, 0.05, stats_txt, transform=ax1.transAxes, bbox=props, ha='left')


ax2.scatter(d,pl,s=20)
ax2.set_xlabel('d [m]',fontsize=12)
ax2.set_ylabel('Measured PL [dB]',fontsize=12)
ax2.plot((0, 1), (0, 1), transform=ax2.transAxes, ls='--',c='k',lw='0.7')

blanks = '\n'.join(['x'*7])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax2.text(0.76, 0.05, blanks, color='none', transform=ax2.transAxes, bbox=props)
# overlay statistics with titles left-aligned and numbers right-aligned
stats_txt = 'MI:0.75'
#stats = '\n'.join([
   # '{}'.format(0.5)
#])
props = dict(boxstyle='square', facecolor='none', edgecolor='none')
ax2.text(0.76, 0.05, stats_txt, transform=ax2.transAxes, bbox=props, ha='left')


ax3.scatter(h,pl,s=20)
ax3.set_xlabel('$\Delta_h$ [m]',fontsize=12)
ax3.set_ylabel('Measured PL [dB]',fontsize=12)
ax3.plot((0, 1), (0, 1), transform=ax3.transAxes, ls='--',c='k',lw='0.7')

blanks = '\n'.join(['x'*7])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax3.text(0.71, 0.05, blanks, color='none', transform=ax3.transAxes, bbox=props)
# overlay statistics with titles left-aligned and numbers right-aligned
stats_txt = 'MI:0.69'
#stats = '\n'.join([
   # '{}'.format(0.5)
#])
props = dict(boxstyle='square', facecolor='none', edgecolor='none')
ax3.text(0.71, 0.05, stats_txt, transform=ax3.transAxes, bbox=props, ha='left')


ax4.scatter(vd,pl,s=20)
ax4.set_xlabel('$v_{depth}$ [m]',fontsize=12)
ax4.set_ylabel('Measured PL [dB]',fontsize=12)
ax4.plot((0, 1), (0, 1), transform=ax4.transAxes, ls='--',c='k',lw='0.7')

blanks = '\n'.join(['x'*7])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax4.text(0.76, 0.05, blanks, color='none', transform=ax4.transAxes, bbox=props)
# overlay statistics with titles left-aligned and numbers right-aligned
stats_txt = 'MI:0.30'
#stats = '\n'.join([
   # '{}'.format(0.5)
#])
props = dict(boxstyle='square', facecolor='none', edgecolor='none')
ax4.text(0.76, 0.05, stats_txt, transform=ax4.transAxes, bbox=props, ha='left')

ax1.tick_params(axis='x', labelsize=11)
ax1.tick_params(axis='y', labelsize=11)
ax2.tick_params(axis='x', labelsize=11)
ax2.tick_params(axis='y', labelsize=11)
ax3.tick_params(axis='x', labelsize=11)
ax3.tick_params(axis='y', labelsize=11)
ax4.tick_params(axis='x', labelsize=11)
ax4.tick_params(axis='y', labelsize=11)

ax1.grid(ls='--')
ax2.grid(ls='--')
ax3.grid(ls='--')
ax4.grid(ls='--')

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.3
                    )

plt.savefig('predictors_pl.eps',format='eps',dpi=1200)
plt.show()
plt.close()

#%%

path=r"Outdoor_data_vegetationdepth_qgis_final.csv"
df = pd.read_csv(path)
df.head()

X = df.iloc[:, [5]] #with vegetation depth -->5 
y = df.iloc[:, [3]]

x_var="numerical"
y_var="numerical"
      
X=X.squeeze()
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

print(mi_value)

#%%

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12,4))
spec = gridspec.GridSpec(ncols=2, nrows=1) 

ax1 = fig.add_subplot(spec[0,0]) 
ax2 = fig.add_subplot(spec[0,1]) 

path=r"R2_final_model_outdoor_2.csv"
df = pd.read_csv(path)
df.head()

#Plot
p1 = max(max(df["y_target"]), max(df["y_target"]))
p2 = min(min(df["y_target"]), min(df["y_target"]))
ax1.plot([p1, p2], [p1, p2], '--',linewidth=1,label= 'Best fit',c='navy')
ax1.scatter(df["y_target"],df["pl_pred_ann"],s=75, label= 'ANN, $R^2 = 0.78$',c='olivedrab',marker='+')
ax1.scatter(df["y_target"],df["pl_pred_svr"],s=55, label= 'SVR, $R^2 = 0.79$',c='darksalmon',marker='D',linewidths=0)
ax1.scatter(df["y_target"],df["pl_pred_rf"],s=55, label= 'RF,  $R^2 = 0.85$',c='turquoise')
ax1.scatter(df["y_target"],df["pl_pred_gtb"],s=55,label= 'GTB, $R^2 = 0.88$',c='crimson',marker='*')
ax1.set_xlabel('Measured path loss [dB]',fontsize=14)
ax1.set_ylabel('Predicted path loss [dB]',fontsize=14)
ax1.set_title('(a) ML based models', fontsize=14)
ax1.grid(ls='--')
ax1.legend()
ax1.set_ylim(102,125)
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)


#Plot
p1 = max(max(df["y_target"]), max(df["y_target"]))
p2 = min(min(df["y_target"]), min(df["y_target"]))
ax2.plot([p1, p2], [p1, p2], '--',linewidth=1,label= 'Best fit',c='navy')
ax2.scatter(df["y_target"],df["pl_pred_abg"],s=75, label= 'ABG, $R^2 = 0.61$',c='olivedrab',marker='+')
ax2.scatter(df["y_target"],df["pl_pred_abg_p"],s=55, label= 'ABG$\Delta_h,v_{depth}$, $R^2 = 0.66$',c='darksalmon',marker='D',linewidths=0)
ax2.scatter(df["y_target"],df["pl_pred_cif"],s=55, label= 'CIF,  $R^2 = 0.54$',c='turquoise')
ax2.scatter(df["y_target"],df["pl_pred_cif_p"],s=55,label= 'CIF$\Delta_h,v_{depth}$, $R^2 = 0.65$',c='crimson',marker='*')
ax2.set_xlabel('Measured path loss [dB]',fontsize=14)
ax2.set_ylabel('Predicted path loss [dB]',fontsize=14)
ax2.set_title('(b) Empirical based models', fontsize=14)
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)

ax2.set_ylim(102,125)
ax2.grid(ls='--')
ax2.legend()


plt.savefig('r2_outdoor.eps',format='eps',dpi=1200)
plt.show()
plt.close()

#%%

x=53-9.5
y=x/1069

A=np.arange(9.5,53,y)
A=A[::-1]

#%%

path=r"SC1_750_Profile_Building.csv"
df = pd.read_csv(path,header=None)
df.head()
