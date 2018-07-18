import cv2
import sys
import argparse
import numpy as np


global bins
    

imag1=cv2.imread(sys.argv[2],0)

def histogram(image):            
    w,h=image.shape[0],image.shape[1]
    mask=np.zeros(256)
    for i in range(0,w):
       for j in range(0,h):
          mask[image[i,j]]+=1

    return mask

histo_gram=histogram(imag1)

def sum_of_the_image(histogram):
    sum_total=0
    for i in range(0,255):
        sum_total+=i*histogram[i]
    return sum_total
        
sum_t=sum_of_the_image(histo_gram)

def pixels(histogram):
    pixel=0
    for i in range(0,255):
        if histogram[i]>0:
            pixel+=histogram[i]
    return pixel

def otsu_threshold(sum_total,histo_gram):
    sum_background=0.0
    sum_foreground=0.0
    weight_background=0.0
    weight_foreground=0.0
    max_var=0.0
    t=0
    total_pixels=pixels(histo_gram)
    for i in range(0,255):
        weight_background+=histo_gram[i]
        if weight_background==0:
            continue
        
        weight_foreground=total_pixels-weight_background
        if weight_foreground==0:
            break
        sum_background+=i*histo_gram[i]
        sum_foreground=sum_total-sum_background
        
        mean_background=sum_background/weight_background
        mean_foreground=sum_foreground/weight_foreground
        class_var=weight_background*weight_foreground*(mean_background-mean_foreground)*(mean_background-mean_foreground)

        if class_var>max_var:
            max_var=class_var
            t=i
    return t

print("threshold",otsu_threshold(sum_t,histo_gram))
def binary_image(image):
    threshold=otsu_threshold(sum_t,histo_gram)
    w,h=image.shape[0],image.shape[1]
    bin_image=np.zeros((w,h))
    for i in range(0,w):
        for j in range(0,h):
            if image[i,j]>=threshold:
                bin_image[i,j]=255
            else:
                bin_image[i,j]=0
    return bin_image

binary=binary_image(imag1)



cv2.imwrite("binary.png",binary)





