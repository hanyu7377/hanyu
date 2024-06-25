# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 08:54:55 2024

@author: HANYUHSIAO
"""

# script2_train_and_evaluate.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def load_data(csv_dir):
    X = []
    y = []
    
    # Mapping labels to integers
    label_mapping = {
        'example-OZO+1.csv': [1, 0, 0],
        'example-OZO+2.csv': [2, 0, 0],
        'example-OZO+3.csv': [3, 0, 0],
        'example-OZO+4.csv': [4, 0, 0],
        'example-OZO+5.csv': [5, 0, 0],
        'example-OZO-1.csv': [-1, 0, 0],
        'example-OZO-2.csv': [-2, 0, 0],
        'example-OZO-3.csv': [-3, 0, 0],
        'example-OZO-4.csv': [-4, 0, 0],
        'example-OZO-5.csv': [-5, 0, 0],
        "example-pla+2.csv": [0, 2, 0],
        "example-pla+4.csv": [0, 4, 0],
        "example-pla+6.csv": [0, 6, 0],
        "example-pla+8.csv": [0, 8, 0],
        "example-pla+10.csv": [0, 10, 0],
        "example-pla+12.csv": [0, 12, 0],
        "example-pla-2.csv": [0, -2, 0],
        "example-pla-4.csv": [0, -4, 0],
        "example-pla-6.csv": [0, -6, 0],
        "example-pla-8.csv": [0, -8, 0],
        "example-pla-10.csv": [0, -10, 0],
        "example-pla-12.csv": [0, -12, 0],
        "example-lpla+2.csv": [0, 0, 2],
        "example-lpla+4.csv": [0, 0, 4],
        "example-lpla+6.csv": [0, 0, 6],
        "example-lpla+8.csv": [0, 0, 8],
        "example-lpla+10.csv": [0, 0, 10],
        "example-lpla+12.csv": [0, 0, 12],
        "example-lpla-2.csv": [0, 0, -2],
        "example-lpla-4.csv": [0, 0, -4],
        "example-lpla-6.csv": [0, 0, -6],
        "example-lpla-8.csv": [0, 0, -8],
        "example-lpla-10.csv": [0, 0, -10],
        "example-lpla-12.csv": [0, 0, -12]
    }
    
    for filename in os.listdir(csv_dir):
        if filename.endswith('.csv') and filename.startswith('example'):
            csv_file = os.path.join(csv_dir, filename)
            
            # Read the CSV into a DataFrame
            df = pd.read_csv(csv_file, header=None)
            
            # Convert DataFrame to numpy array (2D array)
            image = df.values
            
            # Reshape to (height, width, channels) for CNN input
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
            # Append image and label
            y_label = label_mapping.get(filename, [0, 0, 0])  # Get label from mapping, default to [0, 0, 0] if not found
            X.append(image)
            y.append(y_label)
    
    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def train_and_evaluate(X, y, epochs, output_dir, iteration):
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=X.shape[1:]),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),

        #layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        #layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        
        layers.Dense(3)  # Output layer for three values prediction (OZO, leveling1, leveling2)
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])

    # Train the model
    model.fit(X, y, epochs=epochs, batch_size=2, verbose=0)
    
    # Save the model
    model_filename = f"site1_model_iteration_{iteration}.h5"
    model_save_path = os.path.join(output_dir, model_filename)
    model.save(model_save_path)
    print(f"Saved model at {model_save_path}")
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X, y, verbose=0)
    print(f"Training loss: {loss:.4f}, accuracy: {accuracy:.4f}")
    
    # Predictions
    predictions = model.predict(X)
    
    # Calculate R squared for OZO
    ozo_values = np.array([item[0] for item in y])
    r2_ozo = r2_score(ozo_values, predictions[:, 0])
    print(f"R squared (OZO): {r2_ozo:.4f}")
    
    # Calculate R squared for leveling1
    leveling1_values = np.array([item[1] for item in y])
    r2_leveling1 = r2_score(leveling1_values, predictions[:, 1])
    print(f"R squared (leveling1): {r2_leveling1:.4f}")
    
    # Calculate R squared for leveling2
    leveling2_values = np.array([item[2] for item in y])
    r2_leveling2 = r2_score(leveling2_values, predictions[:, 2])
    print(f"R squared (leveling2): {r2_leveling2:.4f}")
    
    # Plot predicted vs true values
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot for OZO
    ax1.scatter(ozo_values, predictions[:, 0], color='blue', label='Predicted vs True (OZO)')
    ax1.plot([-5, 6], [-5, 6], color='red', linestyle='--', label='Ideal Line')
    ax1.set_xlabel('True OZO', fontsize=20)
    ax1.set_ylabel('Predicted OZO', fontsize=20)
    ax1.set_title('True vs Predicted OZO (R^2: {:.2f})'.format(r2_ozo), fontsize=20)
    ax1.legend()
    ax1.grid(False)
    
    # Plot for leveling1
    ax2.scatter(leveling1_values, predictions[:, 1], color='blue', label='Predicted vs True (leveling1)')
    ax2.plot([-12, 12], [-12, 12], color='red', linestyle='--', label='Ideal Line')
    ax2.set_xlabel('True leveling1', fontsize=20)
    ax2.set_ylabel('Predicted leveling1', fontsize=20)
    ax2.set_title('True vs Predicted leveling1 (R^2: {:.2f})'.format(r2_leveling1), fontsize=20)
    ax2.legend()
    ax2.grid(False)
    # Plot for leveling2
    ax3.scatter(leveling2_values, predictions[:, 2], color='blue', label='Predicted vs True (leveling2)')
    ax3.plot([-12, 12], [-12, 12], color='red', linestyle='--', label='Ideal Line')
    ax3.set_xlabel('True leveling2', fontsize=20)
    ax3.set_ylabel('Predicted leveling2', fontsize=20)
    ax3.set_title('True vs Predicted leveling2 (R^2: {:.2f})'.format(r2_leveling2), fontsize=20)
    ax3.legend()
    ax3.grid(False)
    
    plt.tight_layout()
    plt.show()
    
    return r2_ozo, r2_leveling1, r2_leveling2

def plot_box_plots(r2_ozo_results, r2_leveling1_results, r2_leveling2_results):
    # Create a DataFrame for plotting
    df_boxplot = pd.DataFrame({
        'R_squared_OZO': r2_ozo_results,
        'R_squared_leveling1': r2_leveling1_results,
        'R_squared_leveling2': r2_leveling2_results
    })
    
    # Plotting box plots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
    
    # Box plot for R_squared_OZO
    df_boxplot['R_squared_OZO'].plot(kind='box', ax=axes[0])
    axes[0].set_title('R squared (OZO)', fontsize=20)
    axes[0].set_ylabel('R squared value', fontsize=16)
    axes[0].tick_params(axis='both', which='major', labelsize=14)
    
    # Box plot for R_squared_leveling1
    df_boxplot['R_squared_leveling1'].plot(kind='box', ax=axes[1])
    axes[1].set_title('R squared (leveling1)', fontsize=20)
    axes[1].set_ylabel('R squared value', fontsize=16)
    axes[1].tick_params(axis='both', which='major', labelsize=14)
    
    # Box plot for R_squared_leveling2
    df_boxplot['R_squared_leveling2'].plot(kind='box', ax=axes[2])
    axes[2].set_title('R squared (leveling2)', fontsize=20)
    axes[2].set_ylabel('R squared value', fontsize=16)
    axes[2].tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    plt.show()

def main():
    # Directory containing CSV files
    csv_dir = './all test raw image'
    
    # Output directory for models
    output_dir = './output_model'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    X, y = load_data(csv_dir)
    
    # Number of times to repeat training
    num_repeats = 10
    epochs_per_repeat = 20
    
    # Store R squared results
    r2_ozo_results = []
    r2_leveling1_results = []
    r2_leveling2_results = []
    
    # Repeat training process
    for i in range(num_repeats):
        print(f"Training iteration {i + 1}/{num_repeats}")
        r2_ozo, r2_leveling1, r2_leveling2 = train_and_evaluate(X, y, epochs=epochs_per_repeat, output_dir=output_dir, iteration=i + 1)
        r2_ozo_results.append(r2_ozo)
        r2_leveling1_results.append(r2_leveling1)
        r2_leveling2_results.append(r2_leveling2)
    
    # Export R squared results to Excel
    excel_filename = './r_squared_results.xlsx'
    df_results = pd.DataFrame({
        'Iteration': range(1, num_repeats + 1),
        'R_squared_OZO': r2_ozo_results,
        'R_squared_leveling1': r2_leveling1_results,
        'R_squared_leveling2': r2_leveling2_results
    })
    df_results.to_excel(excel_filename, index=False)
    print(f"Exported R squared results to {excel_filename}")
    
    # Plot box plots for R squared values
    plot_box_plots(r2_ozo_results, r2_leveling1_results, r2_leveling2_results)
    
    print("All training iterations completed.")

if __name__ == "__main__":
    main()
