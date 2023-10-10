# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 11:07:19 2021

@author: Siti Hinggit
"""
from __future__ import division
#from skimage import io
#from skimage import feature
#from skimage import data, exposure
#from skimage.color import rgb2hsv
import cv2 
import numpy as np
#from PIL import Image
import glob
#rambutan = ['0_1.jpg', '0_2.jpg', '0_3.jpg',
  #          '1_1.jpg','1_2.jpg', '1_3.jpg', '1_4.jpg', '1_5.jpg', '1_6.jpg', '1_7.jpg','1_.jpg',
#            '2_1.jpg', '2_2.jpg', '2_3.jpg', '2_4.jpg', '2_5.jpg', '2_6.jpg', '2_7.jpg', '2_8.jpg', '2_9.jpg', '2_10.jpg']
path = 'Praproses\*.*'
for bb,file in enumerate (glob.glob(path)):
    #membuat kernel buat nanti masking
    kernelOpen=np.ones((5,5))
    kernelClose=np.ones((20,20))
    
    #baca gambar rgb
    rgb = cv2.imread(file)
   
    # convert BGR to HSV
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
         
    # bikin range buat warna merah 
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
         
    #langsung threshold pake cv.inrange dengan gambarnya ialah hsv
    redmask1 = cv2.inRange(hsv, lower_red, upper_red)
        
    # bikin range lagi buat warna merahnya buat masking kedua kalinya
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
         
    #langsung threshold pake cv.inrange dengan gambarnya ialah hsv
    redmask2 = cv2.inRange(hsv, lower_red, upper_red)
    
    #ngegabungin mask1 sama mask2
    redmask=redmask1+redmask2
    
    #gunain operasi open pake kernel yang tadi diatas
    maskOpen=cv2.morphologyEx(redmask,cv2.MORPH_OPEN,kernelOpen)
    
    #hasil dari open dipake di close ini
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
    
    #hasil akhirnya masking dari close
    maskFinal=maskClose
    
    #setelah itu pake bitwise buat nyatuin gambarnya, cuma pake hsv bukan rgbnya
    maskFinally = cv2.bitwise_and(hsv,hsv, mask = maskFinal)
    cv2.imwrite('Praproses\output_image\masking{}.jpg'.format(bb),maskFinally)