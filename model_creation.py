import cv2
import pandas as pd
import glob
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sparsenet.core import sparse
from sklearn.metrics import precision_recall_fscore_support as score



def model():
    train=pd.read_csv(r"C:\Users\user33\Desktop\neha\code\dataset\emnist-balanced-train.csv")
    train=pd.DataFrame(train)
    
    
    test=pd.read_csv(r"C:\Users\user33\Desktop\neha\code\dataset\emnist-balanced-test.csv")
    test=pd.DataFrame(test)
     
    
    img_test=[]
    label_test=[]
    img_train=[]
    label_train=[]
    
    
    for index, row in train.iterrows():
        row=row["45"]
        label_train.append(row)
        
    
    for index, row in test.iterrows():
        row=row["41"]
        label_test.append(row)
    
    
    no_lable_train=train.drop(['45'], axis = 1)
    no_lable_test=test.drop(['41'], axis = 1)
    
    
    for index, row in no_lable_train.iterrows():
        row=row.values.reshape(28,28)
        row=row.T
        img_train.append(row)
        
        
    for index, row in no_lable_test.iterrows():
        row=row.values.reshape(28,28)
        row=row.T
        img_test.append(row)
        
    
    img_test=np.array(img_test)
    label_test=np.array(label_test)
    img_train=np.array(img_train)
    label_train=np.array(label_train)
    
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16,(3,3),padding='same',activation = "relu" , input_shape = (28,28,1)) ,
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3),padding='same',activation = "relu") ,  
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),padding='same',activation = "relu") ,  
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128,(3,3),padding='same',activation = "relu"),  
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),  
        tf.keras.layers.Dense(1000,activation="relu"),      #Adding the Hidden layer
        tf.keras.layers.Dropout(0.1,seed = 2019),
        tf.keras.layers.Dense(752,activation="relu"),      #Adding the Hidden layer
        tf.keras.layers.Dropout(0.1,seed = 2019),
        tf.keras.layers.Dense(376,activation ="relu"),
        tf.keras.layers.Dropout(0.3,seed = 2019),
        tf.keras.layers.Dense(188,activation="relu"),
        tf.keras.layers.Dropout(0.4,seed = 2019),
        tf.keras.layers.Dense(94,activation ="relu"),
        tf.keras.layers.Dropout(0.2,seed = 2019),
        sparse(60, activation="relu"),#Adding the Output Layer
        tf.keras.layers.Dropout(0.1,seed = 2019),
        tf.keras.layers.Dense(48,activation = "softmax") 
    
    ])
    
    
    opt = tf.keras.optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
    )
    
    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'],)
    model.fit(img_train,
              label_train,
              epochs=10,
              validation_data=(img_test, label_test))
    y_true=label_test
    class_names=[0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,47]
    li=[]
    import time
    start_time = time.time()
    for i in img_test:
        image = cv2.resize(i, (28, 28)) 
        i=image.reshape(-1,28,28)
        y_pred=model.predict(i)
        pred_class = class_names[np.argmax(y_pred)]
        li.append(pred_class)
        print(time.time() - start_time)
    y_pred=np.array(li)

    precision,recall,fscore,support=score(y_true,y_pred,average='macro')
    print ('Precision : {}'.format(precision))
    print ('Recall    : {}'.format(recall))
    print ('F-score   : {}'.format(fscore))
    return model
    