import cv2
import sys
import argparse
import numpy as np


global bins
    
# print(sys.argv[1])
imag1=cv2.imread(sys.argv[2],0)
# blur = cv2.GaussianBlur(imag1,(5,5),0)
def histogram(image):            
    w,h=image.shape[0],image.shape[1]
    mask=np.zeros(255)
    for i in range(0,w):
       for j in range(0,h):
          mask[image[i,j]]+=1

    return mask



##
def sum_of_the_image(histogram):
    sum_total=0.0
    for i in range(0,255):
        sum_total+=i*histogram[i]
    return sum_total
        


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


def binary_image(image):
    threshold=otsu_threshold(sum_total,histo_gram)
    w,h=image.shape[0],image.shape[1]
    bin_image=np.zeros((w,h))
    for i in range(0,w):
        for j in range(0,h):
            if image[i,j]>=threshold:
                bin_image[i,j]=255
            else:
                bin_image[i,j]=0
    return bin_image

global grid_size
grid_size=int(sys.argv[3])
# reconstruct=np.zeros(255)
x=0
y=imag1.shape[1]//grid_size
# print("y",y)
count=0

##histo_gram=histogram(blur)
for r in range(0,grid_size):
        window = imag1[:imag1.shape[0],x:y]
        # print("x,y,imag1.shape",x,y,imag1.shape[0])
        histo_gram=histogram(window)
        sum_total=sum_of_the_image(histo_gram)
        # print("win",window)
        # print("thresh",otsu_threshold(sum_total,histo_gram))
        bin_image=binary_image(window)
        if r==0:
            # print("Vidika")
            cv2.imwrite("binary.png",bin_image)
        else:
            reconstruct=cv2.imread("binary.png",0)
            w,h=reconstruct.shape[0],reconstruct.shape[1]
            w0,h0=bin_image.shape[0],bin_image.shape[1]
            final_image=np.zeros((max(w,w0),h+h0),np.uint8)
            final_image[:w,:h]=reconstruct
            final_image[:w0,h:h+h0]=bin_image
            cv2.imwrite("binary.png",final_image)
        x=y
        y=x+imag1.shape[1]//grid_size
    

       
# print("final_image",bin_image)

        # sum_total=sum_of_the_image(histo_gram)


##        print("thresh",otsu_threshold(sum_total,histo_gram))
        
##        histo_gram = np.histogram(window,bins=255)




# binimage=binary_image(window)
# cv2.imshow("bin",binimage),cv2.waitKey(0)
# print("pix",pixels(histo_gram))
# print("sum_t",sum_of_the_image(histo_gram))


        

