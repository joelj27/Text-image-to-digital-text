import model_creation
import preprocessing_1
from matplotlib import pyplot as plt
import numpy as np  
import tensorflow as tf
import cv2

model=model_creation.model()
p=preprocessing_1.IMVO ()

class_names=[0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,47]
    

output=[]
image_1=[]
    #loop through the list l and save the image 
for i in range(len(p)):
    o=p[i].shape
    print(o[0])
    if o[0] == 0:
        pass
    else:
        j=p[i]
        j = cv2.GaussianBlur(j, (3,3), 2)
        image = cv2.resize(j, (28, 28)) 
        
        #cv2.imshow("y",image)
        j=image.reshape(-1,28,28)
        print(j.shape)
        image_1.append(j)
        yhat = model.predict(j)
        pred_class = class_names[np.argmax(yhat)]
        output.append(pred_class)
                
        print(pred_class)
    
    
        
    
    
dic={0:"\u004f",1:"\u0031",2:"\u0032",3:"\u0033",4:"\u0034",5:"\u0035",6:"\u0036",7:"\u0037",8:"\u0038",9:"\u0039",10:"\u0041",11:"\u0042",12:"\u0043",13:"\u0044",14:"\u0045",15:"\u0046",16:"\u0047",
         17:"\u0048",18:"\u0049",19:"\u004a",20:"\u004b",21:"\u004c",22:"\u004d",23:"\u004e",24:"\u004f",25:"\u0050",26:"\u0051",27:"\u0052",28:"\u0053",29:"\u0054",30:"\u0055",
         31:"\u0056",32:"\u0057",33:"\u0058",34:"\u0059",35:"\u0059",36:"\u0061",37:"\u0062",38:"\u0064",39:"\u0065",40:"\u0066",41:"\u0067",42:"\u0068",43:"\u006e",
         44:"\u0071",45:"\u0072",46:"\u0074",47:"\u0020"}
z=[]
for i in output:
    for key,values in dic.items():
            if i == key:
                print(values)
                z.append(values)
  
w = 10
h = 10
fig = plt.figure(figsize=(20, 20))
columns =10
rows = 10
for i in range(len(image_1)):
    img = np.random.randint(10, size=(h,w))
    f=i+1
    fig.add_subplot(rows, columns, f)  
    d=image_1[i].reshape(28,28)
    r=z[i]
    plt.title(r)
    plt.imshow(d) 
plt.show()

file = open(r"C:\Users\user33\Desktop\neha\output\out.txt", "w+", encoding='utf-8') 
#  the values in the file
for i in z:
    file.write(i)  
file.close( )