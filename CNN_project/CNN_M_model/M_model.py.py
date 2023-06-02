# -*- coding: utf-8 -*-
"""
Created on Mon May 15 15:34:34 2023

@author: hanyu
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout
from tensorflow.keras import layers,regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import seaborn as sns
import random
import time
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import cv2
#from keras-tuner import HyperParameters
##########check my dataframe
total_df = pd.DataFrame()

total_df['images'] = os.listdir("circle_train_val")+os.listdir("square_train_val")+os.listdir("star_train_val")+os.listdir("triangle_train_val") 
index = []
shape_category = []
path = []
count = 0
for i in total_df['images']:
    category = i.split("-")[0]
    shape_category.append(category)
    path.append(str(category)+"_train_val")
total_df['shape_category'] = shape_category
total_df['path'] = path
total_df.isna().sum()
######check my datatframe



######define a nosie function to mess up our patterns
def add_spots_noise(image, prob):
    noisy_image = image.copy()
    height, width, _ = noisy_image.shape
    num_pixels = height * width
    num_spots = int(num_pixels * prob)
    coords = [np.random.randint(0, i - 1, num_spots) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1], :] = np.random.randint(0, 255, (num_spots, 1))

    return noisy_image
######define a nosie function to mess up our patterns






IM_width = 128


#####visulization in pie plot
circle = shape_category.count("circle")
square = shape_category.count("square")
triangle= shape_category.count("triangle")
star = shape_category.count("star")
def get_random_color():
    r = lambda: random.uniform(0,1)
    return [r(),r(),r(),1]

label = ['circle','square','star','triangle']
colors = ["blue","red","yellow","green"] 
#plt.pie([circle,square,triangle,star], labels=label, colors=colors,autopct='%1.2f%%', shadow=False, startangle=0 
#        ,pctdistance=0.7, labeldistance=1.2)
##########visualization in pie plot



#########Load the images and 
def read_all_file_name(shape):
    file_path = f"{shape}_train_val"
    file_names = os.listdir(file_path)
    return file_names

data = []
target = []
shapes = ['circle', 'square', 'star', 'triangle']
intensity = np.arange(0.15,1.501,0.15)

for shape in shapes:
    file_names = read_all_file_name(shape)
    count = 0
    for file_name in file_names:
        
        path = f"{shape}_train_val/{file_name}"
        Im = Image.open(path)
        Ori_image = np.asarray(Im.resize((IM_width , IM_width)))
        Ori_image = Ori_image.reshape(IM_width , IM_width , 1)
        #if count % 2 == 1:
            #Ori_image = add_spots_noise(Ori_image, prob=random.choice(intensity))
            #Ori_image = array_to_img(Ori_image)
            #Ori_image.save(path)
        data.append(Ori_image)
        target.append(shape)
        count +=1
data = np.array(data)
target = np.array(target)
#########Load the images





######plot some of them 
'''
fig = plt.figure(figsize=(20,15))
gs = fig.add_gridspec(6,6 )

for line in range(0, 5):
    for row in range(0, 5):
        num_image = random.randint(0, data.shape[0])
        ax = fig.add_subplot(gs[line, row])
        ax.axis('off');
        ax.set_title(target[num_image],fontsize = 18,fontweight = "bold")
        ax.imshow(data[num_image],cmap = 'gray') 
        
######plot some of them '''


####Let's train the model

X_train, X_val, y_train, y_val = train_test_split(data, np.array(target), test_size=1600, stratify=target,shuffle = True)
X_train_norm = np.round((X_train/255),3 ).copy()
X_val_norm = np.round((X_val/255), 3).copy()


encoder = LabelEncoder().fit(y_train)
y_train_cat = encoder.transform(y_train)
y_val_cat = encoder.transform(y_val)
y_train_oh = to_categorical(y_train_cat)
y_val_oh = to_categorical(y_val_cat)
X_train_norm = X_train_norm.reshape(-1, IM_width ,IM_width  , 1)
X_val_norm = X_val_norm.reshape(-1, IM_width , IM_width , 1)
print(X_train_norm.shape)
start = time.time()
def initialize_model():
    model = Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation="relu",kernel_regularizer=regularizers.l2(0.03), input_shape=(IM_width , IM_width , 1), padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.03),padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.03),padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.03),padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    #model.add(layers.Conv2D(128, (2, 2), activation="relu", kernel_regularizer=regularizers.l2(0.01),padding='same'))
    #model.add(layers.MaxPool2D(pool_size=(2, 2)))
    #model.add(layers.Conv2D(144, (2, 2), activation="relu", kernel_regularizer=regularizers.l2(0.01),padding='same'))
    #model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(rate=0.1))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(rate=0.1))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(rate=0.1))
    model.add(layers.Dense(4, activation='softmax'))
    return model
model = initialize_model()
model.summary()


def compile_model(model):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics="accuracy")
    return model


my_model = initialize_model()
my_model = compile_model(my_model)

history = my_model.fit(X_train_norm, y_train_oh,
                    batch_size=512,
                    epochs=12,   validation_data=(X_val_norm, y_val_oh) )                
                    

print('total time taken this trainning: ', time.time()-start ," second") 
my_model.save('../M_model_result/M_model.h5') 
history_dict = history.history
print(history_dict.keys())



###### summarize history for loss and accuracy
def plot_loss_or_accuracy(a,b,color,title):
    plt.clf()  # 清除所有現有的軸
    plt.cla()  # 清除當前軸的內容
    plt.close()  # 關閉當前的圖形窗口
    line_wid = 4
    fig,ax = plt.subplots(figsize=(13,13))
    plt.plot(history.history[a],marker = 'o',markersize = 20,linewidth = 3,color = color)
    plt.plot(history.history[b],marker = '*',markersize = 20,linewidth = 3,color = color)
    plt.title(title,fontsize = 40,y = 1.05)
    ax.spines['top'].set_linewidth(line_wid)
    ax.spines['bottom'].set_linewidth(line_wid)
    ax.spines['left'].set_linewidth(line_wid)
    ax.spines['right'].set_linewidth(line_wid)
    plt.xlabel('epoch',fontsize = 40)
    plt.tick_params(length = 8 ,width = 4)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.legend(['train_'+ a, b], loc='best',fontsize = 35)
    plt.tight_layout()
    plt.savefig(title,dpi = 800)
    plt.pause(3)

    
plot_loss_or_accuracy('loss','val_loss' , "blue", "Loss plot (M-model)")

plot_loss_or_accuracy('accuracy','val_accuracy' , "orange", "Accuracy plot (M-model)")

###### summarize history for loss and accuracy'''




