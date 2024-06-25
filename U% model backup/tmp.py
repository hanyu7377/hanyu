# tmp.py

import os
import numpy as np
import pandas as pd
from PIL import Image

# Directory containing CSV files
csv_dir = './all test raw image'

# Output directory for processed CSV and images
output_dir = './tmp'
os.makedirs(output_dir, exist_ok=True)

# Function to process each CSV file
def process_csv(csv_file, output_dir):
    # Read CSV into DataFrame
    df = pd.read_csv(csv_file, header=None)
    
    # Convert DataFrame to numpy array (2D array)
    matrix = df.values
    
    # Rotate the matrix 90 degrees clockwise
    rotated_matrix = np.rot90(matrix, k=-1)
    
    # Save rotated matrix to CSV file
    rotated_csv_filename = os.path.basename(csv_file).replace('example-pla', 'example-lpla')
    rotated_csv_path = os.path.join(output_dir, rotated_csv_filename)
    np.savetxt(rotated_csv_path, rotated_matrix, delimiter=",", fmt='%d')
    
    print(f"Saved rotated matrix as {rotated_csv_path}")
    
    # Normalize and convert matrix values to 0-255 scale for image
    normalized_matrix = 255 - (rotated_matrix / np.max(rotated_matrix)) * 255
    normalized_matrix = normalized_matrix.astype(np.uint8)
    
    # Create PIL image from numpy array
    pil_image = Image.fromarray(normalized_matrix)
    
    # Save the PIL image as PNG
    image_filename = os.path.splitext(rotated_csv_filename)[0] + ".png"
    image_save_path = os.path.join(output_dir, image_filename)
    pil_image.save(image_save_path)
    
    print(f"Saved image as {image_save_path}")

# Process each CSV file starting with 'example-plaXX.csv'
for filename in os.listdir(csv_dir):
    if filename.startswith('example-pla') and filename.endswith('.csv'):
        csv_file = os.path.join(csv_dir, filename)
        
        # Process CSV file
        process_csv(csv_file, output_dir)
