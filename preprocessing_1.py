import cv2
import model_creation
from imutils import contours
import numpy as np
import keras
import tensorflow as tf

def IMVO():

    blank=cv2.imread(r'C:\Users\user33\Desktop\CLIENT\neha client 3\code\blank.png')
    blank = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
    
    
    
    image = cv2.imread(r'C:\Users\user33\Desktop\neha\code\input image\IMG_20220108_155152.JPG')
    image = cv2.resize(image, (700, 700))  
    
    
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #blur the image
    blur = cv2.GaussianBlur(gray, (5,5), 1)
    
    
    #set a treshold value
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,29,38)
    
    
    
    #set a kernal value 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100,15))
    
    dilate = cv2.dilate(thresh, kernel, iterations=3)
    
    cv2.waitKey()
    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    l=[]

    #("left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top"):
    (cntts, boundingBoxes) = contours.sort_contours(cnts, method="top-to-bottom")
    
    
    for c in cntts:

                x,y,w,h = cv2.boundingRect(c)
                ROI = image[y:y+h, x:x+w]
    
                l.append(ROI)
    #loop through the list l and save the image 
    for i in range(len(l)):
        cv2.imwrite(f'C:/Users/user33/Desktop/neha/output/line/image{i}.png',l[i])
        
    d=[]
    for i in l:
        i = cv2.resize(i, (500, 100)) 
    
        gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    
    
        #blur the image
        blur = cv2.GaussianBlur(gray, (5,5), 1)
    
        #set a treshold value
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,21)
     
        #set a kernal value 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,7))
    
    
        dilate = cv2.dilate(thresh, kernel, iterations=2)
    
        # Find contours, highlight text areas, and extract ROIs
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        #("left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top"):
        (cntts, boundingBoxes) = contours.sort_contours(cnts, method="top-to-bottom")
        (cnttts, boundingBoxes) = contours.sort_contours(cnts, method="left-to-right")   
            
        for c in cnttts:
            x,y,w,h = cv2.boundingRect(c)
            ROI = i[y:y+h, x:x+w]
            d.append(ROI)
    #loop through the list l and save the image 
    b=[]
    for i in d:
        f=i.shape
        x=f[0]
        print(x)
        if x <11:
            pass
        else:
            b.append(i)
    for i in b:
        print(i.shape)
        
    for i in range(len(b)):
        cv2.imwrite(f'C:/Users/user33/Desktop/neha/output/word/image{i}.png',b[i])
    cv2.waitKey()
    
    p=[]
    for i in b:
        r = cv2.resize(i, (250, 150)) 
        r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
        r = cv2.GaussianBlur(r, (1,3), 1)
        r = cv2.convertScaleAbs(r, alpha=1.5, beta=50)
    
        cv2.waitKey(0) 
        #blur the image
        blur = cv2.GaussianBlur(r, (3,3), 1)
    
    
    
        #set a treshold value
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,61,23)
    
    
        #set a kernal value 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,5))
        cv2.waitKey(0) 
        dilate = cv2.dilate(thresh, kernel, iterations=1)
    
        # Find contours, highlight text areas, and extract ROIs
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        #("left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top"):
    
        (cntts, boundingBoxes) = contours.sort_contours(cnts, method="top-to-bottom")
        (cnttts, boundingBoxes) = contours.sort_contours(cnts, method="left-to-right")   
            
        for c in cnttts:
            x,y,w,h = cv2.boundingRect(c)
            h=h+14
            w=w+14
            x=x-6
            y=y-6
            ROI = dilate[y:y+h, x:x+w]
            p.append(ROI)
        p.append(blank)
        
        for i in range(len(p)):
           cv2.imwrite(f'C:/Users/user33/Desktop/neha/output/letter/image{i}.png',p[i]) 

    
    
    return p