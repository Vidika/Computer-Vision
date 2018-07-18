from __future__ import print_function  
import cv2
import math
import argparse
import numpy as np
import os



def up_to_step_1(imgs):
    modified_imgs=[]
    for img in imgs:
        if img is not None:
            surf = cv2.xfeatures2d.SURF_create(3000)
            kp,descriptors = surf.detectAndCompute(img,None)
            image_changed = cv2.drawKeypoints(img,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            modified_imgs.append(image_changed)
    return modified_imgs


def save_step_1(imgs, output_path='./output/step1'):
    for n, img in enumerate(imgs, start=1):
        m = '0' + str(n) if n < 10 else str(n)
        filename = 'img' + m + '.jpg'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        cv2.imwrite(os.path.join(output_path, filename), img)
    pass


def up_to_step_2(imgs):
    modified_imgs = []
    match_list=[]
    for i in imgs:
        if i is not None:
            surf = cv2.xfeatures2d.SURF_create(3000)
            kp1,descriptor1 = surf.detectAndCompute(i,None)
            for j in imgs:
                if j is not None:
                    kp2,descriptor2 = surf.detectAndCompute(j,None)
                    match_key=Match(descriptor1,descriptor2,2)
                    good=good_matches(match_key)
                    drawParams = dict(matchColor = (0, 0, 255), singlePointColor = (51, 103, 236), matchesMask = good, flags =0)
                    img3 = cv2.drawMatchesKnn(i, kp1, j, kp2, match_key, None, **drawParams)
##                    cv2.imwrite("check.jpg",img3)
                    modified_imgs.append(img3)
                    match_list.append(good)
                


    return modified_imgs,match_list,len(descriptor1),len(descriptor2)      
            
            
            
            
            
    



def save_step_2(imgs, match_list,feature_a,feature_b,output_path="./output/step2"):
    file_name=[]
    for filename in os.listdir(args.input):
        file_name.append(filename)
        
    for n, img in enumerate(imgs, start=1):
        m = '0' + str(n) if n < 10 else str(n)
        filename = '$img'+str(n)+'#'+str(feature_a)+'$img'+str(n+1)+'#'+str(feature_b)+'#'+str(len(match_list))+ '.jpg'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        cv2.imwrite(os.path.join(output_path, filename), img)
    pass



def up_to_step_3(imgs):
    """Complete pipeline up to step 3: estimating homographies and warpings"""
    # ... your code here ...
    return imgs


def save_step_3(img_pairs, output_path="./output/step3"):
    """Save the intermediate result from Step 3"""
    # ... your code here ...
    pass


def up_to_step_4(imgs):
   panorama_image=[]
   for i in range(len(imgs)-1):
       if imgs[i] is not None:
           surf = cv2.xfeatures2d.SURF_create(3000)
           kp1,descriptor1 = surf.detectAndCompute(imgs[i],None)               
           kp2,descriptor2 = surf.detectAndCompute(imgs[i+1],None)
           homography=homography_calculate(kp1,kp2,descriptor1,descriptor2)
           size,offset=size_of_image(imgs[i].shape,imgs[i+1].shape,homography)
           panorama = merge_images(imgs[i], imgs[i+1], homography, size, offset)
           panorama_image.append(panorama)


   return panorama_image
           
                    
def save_step_4(imgs, output_path="./output/step4"):
##    print(imgs)
    for n, img in enumerate(imgs):
        m = '0' + str(n) if n < 10 else str(n)
        filename = 'img' + m + '.jpg'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        cv2.imwrite(os.path.join(output_path, filename), img)
    pass

def Match(descriptor1,descriptor2,k=2):
    match_key=[]
    if len(descriptor1)<len(descriptor2):
        for m in range(len(descriptor1)):
            match_pair=[]
            desc1=descriptor1[m]
            for n in range(len(descriptor2)):
                desc2=descriptor2[n]
                m_desc=cv2.DMatch(m,n,np.linalg.norm(desc1 - desc2, 1, 0))
                match_pair.append(m_desc)                
            match_pair.sort(key=lambda x:x.distance)
            match_key.append(match_pair[:k])
    else:
        for m in range(len(descriptor2)):
            match_pair=[]
            desc2=descriptor2[m]
            for n in range(len(descriptor1)):
                desc1=descriptor1[n]
                m_desc=cv2.DMatch(m,n,np.linalg.norm(desc2 - desc1, 1, 0))
                match_pair.append(m_desc)                
            match_pair.sort(key=lambda x:x.distance)
            match_key.append(match_pair[:k])
##    print("m_key",match_key)
    return match_key
    
        
def good_matches(match_key):
   good_match=[[0,0] for i in range(len(match_key))]
   for i,(m,n) in enumerate(match_key):
       if m.distance < 0.70*n.distance:
           good_match[i]=[1,0]
   return good_match

def homography_calculate(kp1,kp2,des1,des2):
   bf = cv2.BFMatcher()
   matches = bf.knnMatch(des1,des2,k=2)
   good = []
   for m,n in matches:
       if m.distance < 0.75*n.distance:
           good.append(m)

   if len(good)>4:
       source = np.float32([ kp1[m.queryIdx].pt for m in good ])
       destination = np.float32([ kp2[m.trainIdx].pt for m in good ])
       M, mask = cv2.findHomography(source, destination, cv2.RANSAC,5.0)
   return M

def size_of_image(size_image1, size_image2, homography):
  (h1, w1) = size_image1[:2]
  (h2, w2) = size_image2[:2]
  top_left = np.dot(homography,np.array([0,0,1]))
  top_right = np.dot(homography,np.array([w2,0,1]))
  bottom_left = np.dot(homography,np.array([0,h2,1]))
  bottom_right = np.dot(homography,np.array([w2,h2,1]))
  panorama_left = int(min(top_left[0], bottom_left[0], 0))
  panorama_right = int(max(top_right[0], bottom_right[0], w1))
  W = panorama_right - panorama_left 
  panorama_top = int(min(top_left[1], top_right[1], 0))
  panorama_bottom = int(max(bottom_left[1], bottom_right[1], h1))
  H = panorama_bottom - panorama_top
  size = (W, H)
  x = int(min(top_left[0], bottom_left[0], 0))
  y = int(min(top_left[1], top_right[1], 0))
  offset = (-x, -y)


  return size, offset

def merge_images(image1, image2, homography, size, offset):
  (h1, w1) = image1.shape[:2]
  (h2, w2) = image2.shape[:2]
  panorama = np.zeros((size[1], size[0], 3), np.uint8)
  (x, y) = offset
  translation = np.matrix([
    [1.0, 0.0, x],
    [0, 1.0, y],
    [0.0, 0.0, 1.0]
  ])
  homography = np.dot(translation,homography)
  cv2.warpPerspective(image2, homography, size, panorama)
  
  panorama[y:h1+y, x:x+w1] = image1  

  return panorama




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step",
        help="compute image stitching pipeline up to this step",
        type=int
    )

    parser.add_argument(
        "input",
        help="a folder to read in the input images",
        type=str
    )

    parser.add_argument(
        "output",
        help="a folder to save the outputs",
        type=str
    )

    args = parser.parse_args()

    imgs = []
    for filename in os.listdir(args.input):
        print(filename)
        img = cv2.imread(os.path.join(args.input, filename))
        imgs.append(img)

    if args.step == 1:
        print("Running step 1")
        modified_imgs = up_to_step_1(imgs)
        save_step_1(modified_imgs, args.output)
    elif args.step == 2:
##        print("print",imgs)
        print("Running step 2")
##        up_to_step_2(imgs)
        modified_imgs, match_list,feature_a,feature_b = up_to_step_2(imgs)
        save_step_2(modified_imgs, match_list,feature_a,feature_b,args.output)
    elif args.step == 3:
        print("Running step 3")
        up_to_step_3(imgs)
##        save_step_3(img_pairs, args.output)
    elif args.step == 4:
        print("Running step 4")
        img_pairs=up_to_step_4(imgs)
        save_step_4(img_pairs, args.output)
