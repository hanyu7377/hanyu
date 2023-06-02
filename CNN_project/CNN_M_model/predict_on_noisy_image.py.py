# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:14:05 2023

@author: hanyu
"""
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import time
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import seaborn as sns
start = time.time()
import random
import os
import numpy as np
import matplotlib.pyplot as plt

##############lets input the test dataset to see our model performance
test_dir = 'all_test_noisy_only'  
class_names = ['circle', 'square', 'star', 'triangle']
#model = load_model('../C_model_result/C_model.h5')  
model = load_model('../M_model_result/M_model.h5')  

image_paths = []
y_true = []
predictions = []


######define a nosie function to mess up our patterns in testfile
def add_spots_noise(image, prob):
    noisy_image = image.copy()
    height, width, _ = noisy_image.shape
    num_pixels = height * width
    num_spots = int(num_pixels * prob)
    coords = [np.random.randint(0, i - 1, num_spots) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1], :] = np.random.randint(0, 255, (num_spots, 1))

    return noisy_image
######define a nosie function to mess up our patterns in test file
intensity = np.arange(0.15,1.501,0.15)
# Read and preprocess test images
for class_name in class_names:
    class_path = os.path.join(test_dir, class_name)
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        image = load_img(image_path, target_size=(128, 128), color_mode="grayscale")
        image = img_to_array(image)
        
        # Add noise to the image and if you had done that. Please turn off the command at below
        # mage = add_spots_noise(image, prob=random.choice(intensity))        
        # Save the modified image back to the original path
        # modified_image = array_to_img(image)
        # modified_image.save(image_path)
        
        image = np.expand_dims(image, axis=0)
        image = image / 255.0  # Normalize the image
        image_paths.append(image_path)
        y_true.append(class_names.index(class_name))

# Predict each image
for image_path in image_paths:
    image = load_img(image_path, target_size=(128, 128), color_mode="grayscale")
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize the image
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    predictions.append(predicted_class)


cm = confusion_matrix(y_true, predictions)
#thresh = cm.max()/2
# 繪製混淆矩陣
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=True,
            xticklabels=class_names, yticklabels=class_names)
'''for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j,i,format(cm[i,j],"d"),ha="center",va = "center",fontsize = 12)'''
sns.set(font_scale = 2.5)
plt.xticks(fontsize = 24)
plt.yticks(fontsize = 24)
plt.tick_params(length = 8 ,width = 4)
plt.xlabel('Predicted',fontsize = 32,labelpad=16)
plt.ylabel('True',fontsize = 32,labelpad=16)
#plt.title('C-model prediction on noisy images',y=1.1,fontsize = 30)
#plt.savefig("../C_model_result/C model prediction on noisy images",dpi = 800,bbox_inches = 'tight')
plt.title('M-model prediction on noisy images',y=1.1,fontsize = 30)
plt.savefig("../M_model_result/M model prediction on noisy images",dpi = 800,bbox_inches = 'tight')
plt.show()


###########we call back the misclassified_image




misclassified_indices = np.where(np.array(y_true) != np.array(predictions))[0]

num_misclassified_images = len(misclassified_indices)
num_rows = int(np.ceil(num_misclassified_images / 5))
num_columns = 5
fig = plt.figure(figsize=(20, 4 * num_rows))

for i, index in enumerate(misclassified_indices):
    image_path = image_paths[index]
    true_label = class_names[y_true[index]]
    predicted_label = class_names[predictions[index]]
    
    image = load_img(image_path, target_size=(128, 128), color_mode="grayscale")
    
    ax = fig.add_subplot(num_rows, num_columns, i+1)
    ax.imshow(image, cmap='gray')
    ax.set_title(f'True: {true_label}\nPredicted: {predicted_label}',fontsize = 24)
    ax.axis('off')

plt.tight_layout()
plt.savefig("mis_class",dpi = 800,bbox_inches = 'tight')
plt.show()
#########let  calculate the precision, recall and Z1 score in each kind of pattern
tp = [0] * len(class_names)
fp = [0] * len(class_names)
fn = [0] * len(class_names)
for i, index in enumerate(range(len(y_true))):
    true_class = y_true[index]
    pred_class = predictions[index]
    
    if pred_class == true_class:
        tp[true_class] += 1
    else:
        fp[pred_class] += 1
        fn[true_class] += 1
for i, class_name in enumerate(class_names):
    recall = tp[i] / (tp[i] + fn[i]) if tp[i] + fn[i] != 0 else 0
    precision = tp[i] / (tp[i] + fp[i]) if tp[i] + fp[i] != 0 else 0
    Z1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
    
    print(f"Recall（{class_name}）: {recall}")
    print(f"Precision（{class_name}）: {precision}")
    print(f"Z1 score（{class_name}）: {Z1_score}")

