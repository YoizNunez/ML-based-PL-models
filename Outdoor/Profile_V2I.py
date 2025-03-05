# -*- coding: utf-8 -*-
"""
Created on Wed May  3 10:34:38 2023

@author: Yoiz Nuñez
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString

#%%

path=r"SC1_750_Profile_Building.csv"
df_building = pd.read_csv(path,header=None)
df_building.head()

count_values_row = df_building.count(axis='columns')

#%%
path=r"SC1_750_Tx_Rx.csv"
df_tx_rx = pd.read_csv(path)
df_tx_rx.head()

#%%

#Direct Path
Tx= df_tx_rx['Tx'][229]
Rx= df_tx_rx['Rx'][229]
#a=Tx-Rx
#b=a/df_tx_rx['Length'][11]

vector_length=np.linspace(Tx,Rx,df_tx_rx['Length'][229])
vector_distance=np.linspace(1,df_tx_rx['Length'][229],df_tx_rx['Length'][229])

#%%

#reading each profile Tx-Rx
vector_building=np.arange(0,df_tx_rx['Length'][229])

for i in range(df_tx_rx['Length'][229]):

    vector_building[i] = df_building.iloc[229][i]
    
#%%
path=r"SC1_750_Profile_Vegetation.csv"
df_vegetation = pd.read_csv(path,header=None)
df_vegetation.head()


#reading each profile Tx-Rx
vector_vegetation=np.arange(0,df_tx_rx['Length'][229])

for i in range(df_tx_rx['Length'][229]):

    vector_vegetation[i] = df_vegetation.iloc[229][i]
    

#%%
path=r"750_SC1_Profile_Ground.csv"
df_ground = pd.read_csv(path,header=None)
df_ground.head()


#reading each profile Tx-Rx
vector_ground=np.arange(0,df_tx_rx['Length'][229])

for i in range(df_tx_rx['Length'][229]):

    vector_ground[i] = df_ground.iloc[229][i]

#%%
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12,5))
spec = gridspec.GridSpec(ncols=2, nrows=1) 

ax1 = fig.add_subplot(spec[0,0]) 
ax2 = fig.add_subplot(spec[0,1]) 

ax1.plot(vector_distance, vector_length, '--',color='gray',linewidth=1.5)
ax1.plot(vector_distance, vector_building, '-',color='blue',linewidth=1.5)
ax1.plot(vector_distance, vector_vegetation, '-',color='green',linewidth=1.5)
ax1.plot([0,958.35,1348.50,1356.50],[53,43,11,9.40], '-',color='red',linewidth=1)
#ax1.plot(vector_distance, vector_ground, '-',color='black')
ax1.plot(0, 53, 'o',color='red',markersize=7,markeredgecolor='red',markerfacecolor='white')
ax1.plot(1356.51, 9.398839950561523, 'o',color='orange',markersize=8,markeredgecolor='orange',markerfacecolor='white')


ax2.plot(vector_distance, vector_ground, '-',color='black')
ax2.plot(vector_distance, vector_length, '--',color='gray',linewidth=1.5)
ax2.plot(0, 53, 'o',color='red',markersize=7,markeredgecolor='red',markerfacecolor='white')
ax2.plot(1356.51, 9.398839950561523, 'o',color='orange',markersize=8,markeredgecolor='orange',markerfacecolor='white')
ax1.grid(ls='--')
ax2.grid(ls='--')
ax1.set_xlim(-20, 1370)
ax1.set_ylim(0, 55)
ax2.set_xlim(-20, 1370)
ax2.set_ylim(0, 55)
ax1.set_xlabel('Distance [m]',size=12)
ax1.set_ylabel('Heigth [m]',size=12)
ax2.set_xlabel('Distance [m]',size=12)
ax2.set_ylabel('Heigth [m]',size=12)
labels = ['Direct path Tx-Rx', 'Building Profile', 'Vegetation Profile','Difraction component','Tx','Rx','Ground Profile']
fig.legend(labels, loc='center', bbox_to_anchor=(0.5,0.95), ncol=len(labels), bbox_transform=fig.transFigure)
plt.legend()

plt.setp(ax1.get_xticklabels(), fontsize=13)
plt.setp(ax1.get_yticklabels(), fontsize=13)
plt.setp(ax2.get_xticklabels(), fontsize=13)
plt.setp(ax2.get_yticklabels(), fontsize=13)

plt.savefig('profile_Tx_Rx.eps',format='eps',dpi=1200)
plt.show()
plt.close()

#%%
idx = np.argwhere(np.diff(np.sign(vector_length - vector_building))).flatten()

#%%
plt.plot(vector_distance[idx], vector_length[idx], 'ro')
plt.show()

#%%

from shapely.geometry import LineString

first_line = LineString(np.column_stack((vector_distance, vector_length)))
second_line = LineString(np.column_stack((vector_distance, vector_building)))
intersection = first_line.intersection(second_line)

#%%

if intersection.geom_type == 'MultiPoint':
    plt.plot(*LineString(intersection).xy, 'o')
elif intersection.geom_type == 'Point':
    plt.plot(*intersection.xy, 'o')
    
x, y = LineString(intersection).xy


#%%

d_vector=np.arange(0,int(len(idx)/2),dtype=np.float64)
l_vector=np.arange(0,int(len(idx)/2),dtype=np.float64)

a=0
b=1

d=vector_distance[idx]
l=vector_length[idx]

for i in range(int(len(idx)/2)):
    
    d_vector[i] = np.abs(d[i+a]-d[i+b])
    l_vector[i] = np.abs(l[i+a]-l[i+b])
    a+=1
    b+=1
    
#%%

d_3d_vector=np.arange(0,len(d_vector),dtype=np.float64)

for i in range (len(d_vector)):
    d_3d_vector[i] = np.sqrt(d_vector[i]**2+ l_vector[i]**2)
    

#%%

sum_depth_building_2d = sum(d_vector)
sum_depth_building_3d = sum(d_3d_vector)


n_building[i] = len(idx)/2
building_depth_tot_2d[i]= sum_depth_building_2d
building_depth_tot_3d[i]= sum_depth_building_3d


#%%%%%

#BUILDING DEPTH

n_building= np.arange(0,len(df_building))
building_depth_tot_2d= np.arange(0,len(df_building))
building_depth_tot_3d= np.arange(0,len(df_building),dtype=np.float64)
        

for i in range (len(df_building)):
    
    #Direct Path
    Tx= df_tx_rx['Tx'][i]
    Rx= df_tx_rx['Rx'][i]

    vector_length=np.linspace(Tx,Rx,df_tx_rx['Length'][i])
    vector_distance=np.linspace(1,df_tx_rx['Length'][i],df_tx_rx['Length'][i])
        
    #reading each profile Tx-Rx
    vector_building=np.arange(0,df_tx_rx['Length'][i])
    
    for j in range(df_tx_rx['Length'][i]):
    
        vector_building[j] = df_building.iloc[i][j]
        
        
    idx = np.argwhere(np.diff(np.sign(vector_length - vector_building))).flatten()
    
    if len(idx)>1:
        
        first_line = LineString(np.column_stack((vector_distance, vector_length)))
        second_line = LineString(np.column_stack((vector_distance, vector_building)))
        intersection = first_line.intersection(second_line)
        
        if intersection.geom_type == 'MultiPoint':
            plt.plot(*LineString(intersection).xy, 'o')
        elif intersection.geom_type == 'Point':
            plt.plot(*intersection.xy, 'o')
            
        x, y = LineString(intersection).xy
    
    
        d_vector=np.arange(0,int(len(idx)/2),dtype=np.float64)
        l_vector=np.arange(0,int(len(idx)/2),dtype=np.float64)
        
        a=0
        b=1
        
        d=vector_distance[idx]
        l=vector_length[idx]
        
        for k in range(int(len(idx)/2)):
            
            d_vector[k] = np.abs(d[k+a]-d[k+b])
            l_vector[k] = np.abs(l[k+a]-l[k+b])
            a+=1
            b+=1
        
        
        d_3d_vector=np.arange(0,len(d_vector),dtype=np.float64)
        
        for m in range (len(d_vector)):
            d_3d_vector[m] = np.sqrt(d_vector[m]**2+ l_vector[m]**2)
            
        
        sum_depth_building_2d = sum(d_vector)
        sum_depth_building_3d = sum(d_3d_vector)
        
        n_building[i] = len(idx)/2
        building_depth_tot_2d[i]= sum_depth_building_2d
        building_depth_tot_3d[i]= sum_depth_building_3d
    
    
    else:
        
        n_building[i] = 0
        building_depth_tot_2d[i]= 0
        building_depth_tot_3d[i]= 0

#%%

path=r"SC2_3500_Profile_Vegetation.csv"
df_vegetation = pd.read_csv(path,header=None)
df_vegetation.head()

path=r"SC2_3500_Tx_Rx.csv"
df_tx_rx = pd.read_csv(path)
df_tx_rx.head()

#VEGETATION DEPTH
n_vegetation= np.arange(0,len(df_vegetation))
vegetation_depth_tot_2d= np.arange(0,len(df_vegetation))
vegetation_depth_tot_3d= np.arange(0,len(df_vegetation),dtype=np.float64)
        
for i in range (len(df_vegetation)):
    
    #Direct Path
    Tx= df_tx_rx['Tx'][i]
    Rx= df_tx_rx['Rx'][i]

    vector_length=np.linspace(Tx,Rx,df_tx_rx['Length'][i])
    vector_distance=np.linspace(1,df_tx_rx['Length'][i],df_tx_rx['Length'][i])
        
    #reading each profile Tx-Rx
    vector_vegetation=np.arange(0,df_tx_rx['Length'][i])
    
    for j in range(df_tx_rx['Length'][i]):
    
        vector_vegetation[j] = df_vegetation.iloc[i][j]
        
        
    idx = np.argwhere(np.diff(np.sign(vector_length - vector_vegetation))).flatten()
    
    if len(idx)>1:
        
        first_line = LineString(np.column_stack((vector_distance, vector_length)))
        second_line = LineString(np.column_stack((vector_distance, vector_vegetation)))
        intersection = first_line.intersection(second_line)
        
        if intersection.geom_type == 'MultiPoint':
            plt.plot(*LineString(intersection).xy, 'o')
        elif intersection.geom_type == 'Point':
            plt.plot(*intersection.xy, 'o')
            
        x, y = LineString(intersection).xy
    
    
        d_vector=np.arange(0,int(len(idx)/2),dtype=np.float64)
        l_vector=np.arange(0,int(len(idx)/2),dtype=np.float64)
        
        a=0
        b=1
        
        d=vector_distance[idx]
        l=vector_length[idx]
        
        for k in range(int(len(idx)/2)):
            
            d_vector[k] = np.abs(d[k+a]-d[k+b])
            l_vector[k] = np.abs(l[k+a]-l[k+b])
            a+=1
            b+=1
        
        
        d_3d_vector=np.arange(0,len(d_vector),dtype=np.float64)
        
        for m in range (len(d_vector)):
            d_3d_vector[m] = np.sqrt(d_vector[m]**2+ l_vector[m]**2)
            
        
        sum_depth_vegetation_2d = sum(d_vector)
        sum_depth_vegetation_3d = sum(d_3d_vector)
        
        n_vegetation[i] = len(idx)/2
        vegetation_depth_tot_2d[i]= sum_depth_vegetation_2d
        vegetation_depth_tot_3d[i]= sum_depth_vegetation_3d
    
    
    else:
       
        n_vegetation[i] = 0
        vegetation_depth_tot_2d[i]= 0
        vegetation_depth_tot_3d[i]= 0
        
