# -*- coding: utf-8 -*-
"""
Created on Tue May 23 17:46:25 2023

@author: hanyu
"""




import cv2
import numpy as np
import matplotlib.pyplot as plt



def add_spots_noise(image, prob):
    noisy_image = image
    height, width = noisy_image.shape
    num_pixels = height * width
    num_spots = int(num_pixels * prob)
    coords = [np.random.randint(0, i - 1, num_spots) for i in image.shape]
    noisy_image[coords] = np.random.randint(0, 255, num_spots)

    return noisy_image





fig, axes = plt.subplots(2, 5, figsize=(30, 20))
intensity = np.arange(0.15,1.501,0.15)

print(intensity)
for i in range(10):    
    image = cv2.imread('14.png', cv2.IMREAD_GRAYSCALE)
    noisy_image_spots = add_spots_noise(image, prob=intensity[i]) 
    axes[i//5,i%5].imshow(noisy_image_spots, cmap='gray')
    title = 'Spots Noise prob={:.2f}'.format(intensity[i])
    axes[i//5,i%5].set_title(title,fontsize = 20,fontweight = 'bold')
    axes[i//5,i%5].axis('off')
    axes[i//5,i%5].grid(True) 
plt.tight_layout()
plt.savefig('noise display',bbox_inches = "tight",dpi = 800)
plt.show()