# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 17:15:03 2022

@author: KIIT
"""
import numpy as np
import math
import cv2
import csv
daylist = [["Day", "Hue"]]
for j in range(1,6):
     for i in range(20):
         path = "images/Day" + str(j) +"/Day" + str(j) + "_" + str(i) +".jpg"
         image = cv2.imread(path)
         color = image[300,300]
         region= image[0:299,0:299]
         b,g,r = np.mean(region, axis=(0,1))
         r_ = r/255
         g_ = g/255
         b_ = b/255
         Cmax = max(r_,g_,b_)
         Cmin = min(r_,g_,b_)
         delta = Cmax-Cmin
         H=0
         # Hue Calculation
         if delta == 0:
             H = 0
         elif Cmax == r_ :
             H = 60 * (((g_ - b_)/delta) % 6)
         elif Cmax == g_:
             H = 60 * (((b_ - r_)/delta) + 2)
         elif Cmax == b_:
             H = 60 * (((r_ - g_)/delta) + 4)
         print(H)
         daylist.append(["Day"+str(j), H])
with open('HueValue.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(daylist)