# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:50:18 2024

@author: HANYUHSIAO
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras import models

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

    return image

def load_random_model(model_dir):
    # List all files in the output_model directory
    model_files = os.listdir(model_dir)
    
    # Filter out non-model files (assuming model files have .h5 extension)
    model_files = [file for file in model_files if file.endswith('.h5')]
    print(model_files)
    # Choose a random model file
    if len(model_files) > 0:
        chosen_model = model_files[2]
        print(chosen_model)
        model_path = os.path.join(model_dir, chosen_model)
        return model_path
    else:
        raise ValueError("No model files found in the specified directory.")

def main():
    # Directory containing the .csv file for prediction
    csv_file = './prediction/example.csv'
    
    # Output directory for processed .csv files
    output_dir = './processed_csv'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the .csv file and obtain the image matrix
    image_matrix = process_csv(csv_file, output_dir)
    
    # Directory containing trained models
    model_dir = './output_model'
    
    # Load a random model from the model directory
    model_path = load_random_model(model_dir)
    print(f"Using model: {model_path}")
    
    # Load the model
    model = models.load_model(model_path)
    
    # Reshape image matrix to (1, grid_size, grid_size, 1) for prediction
    input_image = image_matrix.reshape(1, image_matrix.shape[0], image_matrix.shape[1], 1)
    
    # Perform prediction
    predictions = model.predict(input_image)
    predictions[0][0] *= -1
    predictions[0][1] *= -1
    predictions[0][2] *= 1
    print(f"Predicted values: {predictions[0]}")

if __name__ == "__main__":
    main()