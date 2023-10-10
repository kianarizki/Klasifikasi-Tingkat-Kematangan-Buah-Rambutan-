# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 12:38:26 2021

@author: Siti Hinggit
"""

from __future__ import division
import numpy as np
import cv2 as cv
#import matplotlib as plt
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
#ubah ke histogram 
#kelas 0 belum matang
masking0_hist,bins = np.histogram((cv.imread('Praproses\output_image\masking0.jpg')).ravel(),256,[0,256])
masking1_hist,bins = np.histogram((cv.imread('Praproses\output_image\masking1.jpg')).ravel(),256,[0,256])
masking2_hist,bins = np.histogram((cv.imread('Praproses\output_image\masking2.jpg')).ravel(),256,[0,256])

#kelas 1 setengah matang
masking3_hist,bins = np.histogram((cv.imread('Praproses\output_image\masking3.jpg')).ravel(),256,[0,256])
masking4_hist,bins = np.histogram((cv.imread('Praproses\output_image\masking4.jpg')).ravel(),256,[0,256])
masking5_hist,bins = np.histogram((cv.imread('Praproses\output_image\masking5.jpg')).ravel(),256,[0,256])
masking6_hist,bins = np.histogram((cv.imread('Praproses\output_image\masking6.jpg')).ravel(),256,[0,256])
masking7_hist,bins = np.histogram((cv.imread('Praproses\output_image\masking7.jpg')).ravel(),256,[0,256])
masking8_hist,bins = np.histogram((cv.imread('Praproses\output_image\masking8.jpg')).ravel(),256,[0,256])
masking9_hist,bins = np.histogram((cv.imread('Praproses\output_image\masking9.jpg')).ravel(),256,[0,256])
masking10_hist,bins = np.histogram((cv.imread('Praproses\output_image\masking10.jpg')).ravel(),256,[0,256])

#kelas 2 matang
masking11_hist,bins = np.histogram((cv.imread('Praproses\output_image\masking11.jpg')).ravel(),256,[0,256])
masking12_hist,bins = np.histogram((cv.imread('Praproses\output_image\masking12.jpg')).ravel(),256,[0,256])
masking13_hist,bins = np.histogram((cv.imread('Praproses\output_image\masking3.jpg')).ravel(),256,[0,256])
masking14_hist,bins = np.histogram((cv.imread('Praproses\output_image\masking14.jpg')).ravel(),256,[0,256])
masking15_hist,bins = np.histogram((cv.imread('Praproses\output_image\masking15.jpg')).ravel(),256,[0,256])
masking16_hist,bins = np.histogram((cv.imread('Praproses\output_image\masking16.jpg')).ravel(),256,[0,256])
masking17_hist,bins = np.histogram((cv.imread('Praproses\output_image\masking17.jpg')).ravel(),256,[0,256])
masking18_hist,bins = np.histogram((cv.imread('Praproses\output_image\masking18.jpg')).ravel(),256,[0,256])
masking19_hist,bins = np.histogram((cv.imread('Praproses\output_image\masking19.jpg')).ravel(),256,[0,256])
masking20_hist,bins = np.histogram((cv.imread('Praproses\output_image\masking20.jpg')).ravel(),256,[0,256])

#transpose + normalisai
masking0_hist = masking0_hist/np.sum(masking0_hist)
masking0_hist_n = np.transpose(masking0_hist[0:34,np.newaxis])
masking1_hist=masking1_hist/np.sum(masking1_hist)
masking1_hist_n = np.transpose(masking1_hist[0:34,np.newaxis])
masking2_hist=masking2_hist/np.sum(masking2_hist)
masking2_hist_n = np.transpose(masking2_hist[0:34,np.newaxis])
masking3_hist=masking3_hist/np.sum(masking3_hist)
masking3_hist_n = np.transpose(masking3_hist[0:34,np.newaxis])
masking4_hist=masking4_hist/np.sum(masking4_hist)
masking4_hist_n = np.transpose(masking4_hist[0:34,np.newaxis])
masking5_hist=masking5_hist/np.sum(masking5_hist)
masking5_hist_n = np.transpose(masking5_hist[0:34,np.newaxis])
masking6_hist=masking6_hist/np.sum(masking6_hist)
masking6_hist_n = np.transpose(masking6_hist[0:34,np.newaxis])
masking7_hist=masking7_hist/np.sum(masking7_hist)
masking7_hist_n = np.transpose(masking7_hist[0:34,np.newaxis])
masking8_hist=masking8_hist/np.sum(masking8_hist)
masking8_hist_n = np.transpose(masking8_hist[0:34,np.newaxis])
masking9_hist=masking9_hist/np.sum(masking9_hist)
masking9_hist_n = np.transpose(masking9_hist[0:34,np.newaxis])
masking10_hist=masking10_hist/np.sum(masking10_hist)
masking10_hist_n = np.transpose(masking10_hist[0:34,np.newaxis])
masking11_hist=masking11_hist/np.sum(masking11_hist)
masking11_hist_n = np.transpose(masking11_hist[0:34,np.newaxis])
masking12_hist=masking12_hist/np.sum(masking12_hist)
masking12_hist_n = np.transpose(masking12_hist[0:34,np.newaxis])
masking13_hist=masking13_hist/np.sum(masking13_hist)
masking13_hist_n = np.transpose(masking13_hist[0:34,np.newaxis])
masking14_hist=masking14_hist/np.sum(masking14_hist)
masking14_hist_n = np.transpose(masking14_hist[0:34,np.newaxis])
masking15_hist=masking15_hist/np.sum(masking15_hist)
masking15_hist_n = np.transpose(masking15_hist[0:34,np.newaxis])
masking16_hist=masking16_hist/np.sum(masking16_hist)
masking16_hist_n = np.transpose(masking16_hist[0:34,np.newaxis])
masking17_hist=masking17_hist/np.sum(masking17_hist)
masking17_hist_n = np.transpose(masking17_hist[0:34,np.newaxis])
masking18_hist=masking18_hist/np.sum(masking18_hist)
masking18_hist_n = np.transpose(masking18_hist[0:34,np.newaxis])
masking19_hist=masking19_hist/np.sum(masking19_hist)
masking19_hist_n = np.transpose(masking19_hist[0:34,np.newaxis])
masking20_hist=masking20_hist/np.sum(masking20_hist)
masking20_hist_n = np.transpose(masking20_hist[0:34,np.newaxis])

#menggabungkan data menjadi satu matriks train
#'''
dataLatih = np.concatenate((masking0_hist_n, masking1_hist_n, masking2_hist_n,
                            masking3_hist_n, masking4_hist_n, masking5_hist_n, 
                            masking6_hist_n, masking7_hist_n, masking8_hist_n, 
                            masking9_hist_n, masking10_hist_n, masking11_hist_n,
                            masking12_hist_n, masking13_hist_n, masking14_hist_n,
                            masking15_hist_n, masking16_hist_n, masking17_hist_n,
                            masking18_hist_n, masking19_hist_n, masking20_hist_n),axis = 0).astype(np.float32)


#memmberi label
label = np.array([0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2]).astype(np.float32)

from sklearn.model_selection import StratifiedKFold
knn = KNeighborsClassifier(n_neighbors=7)
skf = StratifiedKFold(n_splits = 4)
mean = 0
for train, test in skf.split(dataLatih,label) : 
      X_latih, X_uji, Y_latih, Y_uji, = dataLatih[train], dataLatih[test], label[train], label[test]
      knn.fit(X_latih, Y_latih)
      y_pred = knn.predict(X_uji)
      print(accuracy_score(Y_uji, y_pred))
      print(confusion_matrix(Y_uji, y_pred, labels = [0,1,2]) )

      
#knn=KNeighborsClassifier(n_neighbors=5)
akurasi = cross_val_score(knn, dataLatih, label, cv=4, scoring='accuracy')
print('akurasi : ', akurasi.mean())
