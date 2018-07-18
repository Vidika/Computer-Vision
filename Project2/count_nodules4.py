import cv2
import numpy as np
import sys
import argparse

image=cv2.imread(sys.argv[2],0)
# area=int(sys.argv[4])
connect=4
width=image.shape[0]
height=image.shape[1]
final_image=np.zeros((width,height))
label=1

global bins
    
##print(sys.argv[1])
##imag1=cv2.imread(sys.argv[1],0)
##blur = cv2.GaussianBlur(imag1,(5,5),0)
def histogram(image):            
    w,h=image.shape[0],image.shape[1]
    mask=np.zeros(256)
    for i in range(0,w):
       for j in range(0,h):
          mask[image[i,j]]+=1

    return mask

histo_gram=histogram(image)

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

# print("threshold",otsu_threshold(sum_t,histo_gram))
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

binary=binary_image(image)

def neighbours_1(image,i,j):
    labels=[]
    west=image[i][j-1]
    if west.all()!=0:
        labels.append(int(image[i][j-1]))
        # print(image[i][j-1])

    return labels

def neighbours_2(image,i,j):
    labels=[]
    north=image[i-1][j]
    if north.all()!=0:
        labels.append(int(image[i-1][j]))

    return labels



def nodule(binary):
    count=1
    components=[[]]
    for i in range(len(binary)):
        for j in range(len(binary[i])):
            if binary[i][j]==0:
                label=[]
                if j:
                    labels=neighbours_1(final_image,i,j)
                    if len(labels)!=0:
                        label.append(labels[0])
                        
                if i:
                    labels=neighbours_2(final_image,i,j)
                    if len(labels)!=0:
                        label.append(labels[0])
                if len(label)==0:
                    final_image[i][j]=count
                    count+=1
                
                elif len(label)==1:
                    # print(label[0])
                    final_image[i][j]=label[0]
                else:
                    final_image[i][j]=min(label)
                    for k,x in enumerate(components):
                        if len(x)==0:
                            components[k].extend(set(label))
                            components.append([])
                            break
                        if len(list(set(components[k])& set(label)))!=0:
                                components[k]=list(set(label).union(set(x)))
                                break
    





    return final_image,components


final_image,components=nodule(binary)
# print('final',final_image)
# print('comp',components)

def connection(final_image,components):
    for i in range(len(final_image)):
        for j in range(len(final_image[i])):
            if final_image[i][j]!=0:
                for k in (components):
                    if final_image[i][j] in k:
                        final_image[i][j]=min(k)

    return final_image
                    

def objects(final_image):
    connected={}
    a=0
    for i in range(len(final_image)):
        for j in range(len(final_image[i])):
            if int(final_image[i][j])>a:
                a=int(final_image[i][j])
    for i in range(1,a+1):
        connected[i]=0
    for i in range(len(final_image)):
        for j in range(len(final_image[i])):
            if final_image[i][j]!=0:
                connected[int(final_image[i][j])]=connected[int(final_image[i][j])]+1
    

    return connected






array=connection(final_image,components)
object_in_image=objects(array)
# print(object_in_image)
max_value = max(object_in_image.values())
max_keys = [k for k, v in object_in_image.items() if v == max_value]
print("The label {} has {} many pixels.".format(max_keys,max_value))



			












	
	
# print("final",final_image)

# def 4_nodule()
