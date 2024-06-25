# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 08:54:27 2024

@author: HANYUHSIAO
"""

import pandas as pd
import numpy as np
from PIL import Image
import os

def process_csv(csv_file, output_dir):
    # Read CSV into DataFrame
    df = pd.read_csv(csv_file)
    
    # Extract X, Y, and THK values
    X = df['X'].values
    Y = df['Y'].values
    THK = df['THK'].values
    
    # Define grid size
    grid_size = 36
    
    # Create 2D grid for the image
    image = np.zeros((grid_size, grid_size))
    
    # Z-score normalization for thickness values
    min_THK = np.min(THK)
    normalized_THK = THK - min_THK + 50
    
    # Assign thickness values to corresponding grid cells
    for i in range(len(X)):
        x_idx = int((X[i] - np.min(X)) / (np.max(X) - np.min(X)) * (grid_size - 1))
        y_idx = int((Y[i] - np.min(Y)) / (np.max(Y) - np.min(Y)) * (grid_size - 1))
        image[y_idx, x_idx] = normalized_THK[i]
    
    csv_filename = os.path.splitext(os.path.basename(csv_file))[0] + ".csv"
    csv_save_path = os.path.join(output_dir, csv_filename)
    
    # Save the matrix to .csv file without normalization
    np.savetxt(csv_save_path, image, delimiter=",", fmt='%d')
    
    # Normalize and convert image values to 0-255 scale
    image = 255 - (image / np.max(image)) * 255
    image = image.astype(np.uint8)
    
    # Create PIL image from numpy array
    pil_image = Image.fromarray(image)
    
    # Save the PIL image as PNG
    image_filename = os.path.splitext(os.path.basename(csv_file))[0] + ".png"
    image_save_path = os.path.join(output_dir, image_filename)
    pil_image.save(image_save_path)
    
    return csv_save_path, image_save_path

def main():
    # Directory containing CSV files
    csv_dir = './all test raw'
    
    # Output directory for processed files
    output_dir = './all test raw image'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each CSV file
    for filename in os.listdir(csv_dir):
        if filename.endswith('.csv'):
            csv_file = os.path.join(csv_dir, filename)
            process_csv(csv_file, output_dir)
    
    print("Processing complete.")

if __name__ == "__main__":
    main()