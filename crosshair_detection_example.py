import numpy as np
import matplotlib.pyplot as plt
import crosshair_definitions as ch
import cv2 
import scipy.optimize as opt



# open images ###############################################
a = 2
b = 4

image_range = np.array([a,b]) #filename range of images
image_count = np.linspace(a,b,b-a+1)
center_coordinates = np.zeros((2*(b-a+1),3)) #collect center coordinates for all images

for image_i in image_count:
    path = "IMG_00%.2d.jpg"%image_i
    im = ch.imread(path)
    print(path,' opened')       
#detect lines ###############################################
    fit_lines,center1,center2 = ch.find2Xhair(im)
    cx1,cy1=center1
    cx2,cy2=center2
# draw lines and center ####################################   
    ch.drawlines(fit_lines,im,1)
    cv2.circle(im,(int(cx1),int(cy1)),15,(255,0,0),5)    
    cv2.circle(im,(int(cx2),int(cy2)),15,(0,255,0),5)    
    
# save to txt ####################################
    center_coordinates[2*(int(image_i)-a),0]=cx1
    center_coordinates[2*(int(image_i)-a),1]=cy1
    center_coordinates[2*(int(image_i)-a),2]=image_i+0.1                     

    center_coordinates[2*(int(image_i)-a)+1,0]=cx2
    center_coordinates[2*(int(image_i)-a)+1,1]=cy2
    center_coordinates[2*(int(image_i)-a)+1,2]=image_i+0.2 
                       
# save data ####################################
    cv2.imwrite("IMG_00%.2d_lines.png"%image_i,im)
    print(path,' saved')
np.savetxt('center_coordinates.txt',center_coordinates,fmt="%s", delimiter=' ') #save txt-file
