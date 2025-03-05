# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:29:35 2023

@author: Yoiz Nuñez
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image
from matplotlib import image as mpimg
import os
img1 = matplotlib.image.imread(r'C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\GE\List of Images\0.jpg')
img2 = matplotlib.image.imread(r'C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\GE\List of Images\1.jpg')

b_img1 = img1[:,:,0]
b_img2 = img2[:,:,0]
g = img2[:,:,1]
r = img2[:,:,2]

img_final = np.abs(b_img1-b_img2)
plt.imshow(img_final)

#img=mpimg.imread(r'C:\Users\Yoiz Nuñez\Documents\DOUTORADO 2023\GE\List of Images\0.jpg')

#%%

from PIL import Image
im = Image.fromarray(new_img)
im.save("Heigth.jpg")

#%%

plt.imshow(new_img)

#%%

new_img=np.zeros((768,1024),dtype=np.uint8)
flag_array=np.zeros((768,1024),dtype=np.uint8)
flag_array2=np.zeros((768,1024),dtype=np.uint8)

#%%

for i in range(49): #total of images

    if i>0:
        
        #imagen anterior
        base_dir=r'C:/Users/Yoiz Nuñez/Documents/DOUTORADO 2023/GE/List of Images/'
        filename=str(i-1)+'.jpg'
        fullpath = os.path.join(base_dir, filename)
        img_anterior=matplotlib.image.imread(fullpath)
        
        #imagen actual
        base_dir=r'C:/Users/Yoiz Nuñez/Documents/DOUTORADO 2023/GE/List of Images/'
        filename=str(i)+'.jpg'
        fullpath = os.path.join(base_dir, filename)
        img_actual=matplotlib.image.imread(fullpath)
        
        b_img_anterior = img_anterior[:,:,0] #banda b de la imagen
        b_img_actual = img_actual[:,:,0]
        
        
        for x in range(767): #numero de pixel en x
            for y in range(1023): #numero de pixel en y
                
                pix_value=np.abs(b_img_anterior[x][y] - b_img_actual[x][y])
                
                if flag_array[x][y]==0:
                    new_img[x][y]=pix_value
                   
                    
                if pix_value>0 and flag_array2[x][y]==0:
                    new_img[x][y]=i
                    flag_array[x][y]=1
                    flag_array2[x][y]=1
       

