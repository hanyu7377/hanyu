import numpy as np
import pandas as pd
import shutil 
import os
import random
####隨機選取2000張照片並將他們拆成 test data, validation date, test data
def create_folder_for_training(shape): 
   if not os.path.isdir(shape):
       os.mkdir(shape)       
create_folder_for_training("circle_train_val")
create_folder_for_training("square_train_val")
create_folder_for_training("triangle_train_val")
create_folder_for_training("star_train_val") 
def creat_folder_for_test(shape):
    if not os.path.isdir("all_test"):
        os.mkdir("all_test")       
    os.chdir('all_test')
    if not os.path.isdir(shape):        
        os.mkdir(shape)
    os.chdir("../")
creat_folder_for_test('circle')    
creat_folder_for_test('square') 
creat_folder_for_test('triangle') 
creat_folder_for_test('star')     
def move_figure(shape):
    pathdir = os.listdir(shape)
    print(pathdir)
    filenumber = 3000
    test = []
    train_val = []   
    sample = random.sample(pathdir,filenumber)
    print(sample)
    for i in range(filenumber):
        if i < 2000:  
           random_figure = random.choice(sample)
           train_val.append(random_figure)       
           shutil.copy(str(shape) + "/" +str(random_figure),str(shape)+"_train_val/" + str(shape)+ "-" + str(random_figure))
           sample.remove(random_figure)
        elif i>=2000:
           random_figure = random.choice(sample)
           test.append(random_figure)
           shutil.copy(str(shape) + "/" +str(random_figure),'all_test/'+ str(shape)+'/'+ str(shape)+ "-"+str(random_figure))
           sample.remove(random_figure)        
    return   
move_figure("circle")
move_figure("square")
move_figure("triangle")
move_figure("star")





