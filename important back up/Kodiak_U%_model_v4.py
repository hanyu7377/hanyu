# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 14:52:18 2024

@author: HANYUHSIAO
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 17:06:21 2024

@author: HANYUHSIAO
"""

import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from tkinter import messagebox
from tkinter import ttk
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import math

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg

print("Starting the application...")
tkdnd_path = r'C:\ProgramData\anaconda3\tcl\tkdnd2.8'
os.environ['TKDND_LIBRARY'] = tkdnd_path

import itertools

from scipy.interpolate import griddata
import os




################畫圖的部分
def process_data_for_plot(data,csv_file):
    df = pd.read_csv(csv_file)  # 替換為你的數據文件路徑
    my_df = df[["X", "Y", "radius", data]].copy()

# 將 radius 進行四捨五入到小數點後一位
    my_df['rounded_radius'] = my_df['radius'].round(1)

# 計算 Data 1 的平均值
    Data_1_mean = df[data].mean()

# 基於 rounded_radius 對 Data 1 進行分組聚合，計算 sum 和 mean
    grouped = my_df.groupby('rounded_radius')[data].agg(['sum', 'mean']).reset_index()

# 重命名聚合後的欄位
    grouped.rename(columns={'sum': 'grouped_sum', 'mean': 'grouped_mean'}, inplace=True)

# 將聚合結果與原始的 my_df 合併
    my_df = pd.merge(my_df, grouped, on='rounded_radius', how='left')

# 新增 "Radial" 欄位，計算 Data 1 全體平均減去 grouped_mean
    my_df['Radial'] = my_df['grouped_mean'] - Data_1_mean 

# 計算 THK
    my_df["Test_THK"] = my_df[data] - my_df["Radial"]

# 筛选 rounded_radius 为 145 的数据
    filtered_df = my_df[my_df['rounded_radius'] == 145]

# 生成有效的点对
    point_pairs = list(itertools.combinations(filtered_df[['X', 'Y']].values, 2))

# 创建一个列表，用于存储符合条件的点对
    valid_pairs = []
    for (x1, y1), (x2, y2) in point_pairs:
        if np.isclose(x1, -x2) and np.isclose(y1, -y2):
            valid_pairs.append(((x1, y1), (x2, y2)))

# 创建一个新的 DataFrame 来存储中间计算结果
    tmp_df = my_df.copy()

# 初始化一个列表来存储所有要输出的列名
    additional_columns = []

# 定义一个函数来判断点 (X, Y) 在给定线的哪一侧
    def determine_side(X, Y, X1, Y1, X2, Y2):
        det = (X - X1) * (Y2 - Y1) - (Y - Y1) * (X2 - X1)
        if det > 0:
            return '1'
        elif det < 0:
            return '2'
        else:
            return 'on the line'

# 初始化一个空的列来存储类别
    for i, ((x1, y1), (x2, y2)) in enumerate(valid_pairs):
        column_name = f'category_{i+1}'
        tmp_df[column_name] = tmp_df.apply(lambda row: determine_side(row['X'], row['Y'], x1, y1, x2, y2), axis=1)
    
    # 为每一条线的分类结果计算 Test_THK 的平均值
        grouped_category = tmp_df.groupby(column_name)[data].mean().reset_index()
        print(grouped_category)
        grouped_category.columns = [column_name, f'Test_THK_mean_{i+1}']
        tmp_df = pd.merge(tmp_df, grouped_category, on=column_name, how='left')
    
    # 将该条线的列名添加到 additional_columns 列表中，确保列的顺序
        additional_columns.append(column_name)
        additional_columns.append(f'Test_THK_mean_{i+1}')

# 输出最终的列顺序：先输出原始 my_df 的列，再输出新增的列
    final_columns = list(my_df.columns) + additional_columns

# 按照设定好的顺序重新排列列并保存到 CSV
    tmp_df = tmp_df[final_columns]
#tmp_df.to_csv('processed_data_with_all_categories_2.csv', index=False)

# 初始化一个字典，用于存储每个 Test_THK_mean 的 range
    range_dict = {}


# 对所有 Test_THK_mean 列进行处理
    for i in range(1, 13):
        category_col = f'category_{i}'
        thk_col = f'Test_THK_mean_{i}'

    # 过滤掉 category 为 "on the line" 的行
        filtered_df = tmp_df[tmp_df[category_col] != "on the line"]

        if not filtered_df.empty:
        # 计算当前 Test_THK_mean 的 range
            current_range = filtered_df[thk_col].max() - filtered_df[thk_col].min()
            range_dict[i] = current_range
# 找出 range 最大的 Test_THK_mean 对应的 index
    max_range_index = max(range_dict, key=range_dict.get)

# 输出最大 range 对应的 Test_THK_mean 列和它的 range
    max_range_col = f'Test_THK_mean_{max_range_index}'
    max_range_value = range_dict[max_range_index]

    print(f'Max range is for {max_range_col} with a value of {max_range_value}')

# 找到对应 category 是 "on the line" 的点
    on_the_line_df = tmp_df[tmp_df[f'category_{max_range_index}'] == "on the line"]




# 确保有至少两个点在 "on the line" 上
    if len(on_the_line_df) < 2:
        raise ValueError("Not enough points on the line to form a line.")

# 使用前两个 "on the line" 的点来确定直线
    x1, y1 = on_the_line_df.iloc[0][['X', 'Y']]
    x2, y2 = on_the_line_df.iloc[1][['X', 'Y']]

# 定义函数来计算点 (X, Y) 到给定直线的距离
    def calculate_distance_to_line(X, Y, X1, Y1, X2, Y2):
        return np.abs((Y2 - Y1) * X - (X2 - X1) * Y + X2 * Y1 - Y2 * X1) / np.sqrt((Y2 - Y1)**2 + (X2 - X1)**2)

# 计算所有点到直线的距离，并乘上 Test_THK_mean 除以 145
    tmp_df['Planar'] = tmp_df.apply(
        lambda row: 0 if row[f'category_{max_range_index}'] == "on the line" 
        else calculate_distance_to_line(row['X'], row['Y'], x1, y1, x2, y2) * max_range_value / 145, 
        axis=1
)

# 过滤掉 category 为 "on the line" 的行
    filtered_tmp_df = tmp_df[tmp_df[f'category_{max_range_index}'] != "on the line"]

# 取得 Test_THK_mean 列的最小值和最大值
    min_thk_value = filtered_tmp_df[max_range_col].min()
    #max_thk_value = filtered_tmp_df[max_range_col].max()

# 对 Planar 进行修正，如果对应的 Test_THK_mean 等于最小值，则乘以 -1
    tmp_df['Planar'] = tmp_df.apply(
        lambda row: -row['Planar'] if row[max_range_col] == min_thk_value else row['Planar'], 
        axis=1
        )

    tmp_df["Residual"] = tmp_df[data] - tmp_df["Radial"] - tmp_df["Planar"]


# 保存最终结果到 CSV 文件
    tmp_df.to_csv('final_processed_data_with_all.csv', index=False)
    return tmp_df

#print(process_data_for_plot("side1"))
#print(process_data_for_plot("side1")["Radial"])




###########這邊把THK轉成矩陣然後把矩陣丟給模型預測
def process_csv(df):
    X = df['X'].values
    Y = df['Y'].values
    THK = df['THK'].values
    
    grid_size = 36
    image = np.zeros((grid_size, grid_size))
    
    min_THK = np.min(THK)
    normalized_THK = THK - min_THK + 50
    
    for i in range(len(X)):
        x_idx = int((X[i] - np.min(X)) / (np.max(X) - np.min(X)) * (grid_size - 1))
        y_idx = int((Y[i] - np.min(Y)) / (np.max(Y) - np.min(Y)) * (grid_size - 1))
        image[y_idx, x_idx] = normalized_THK[i]
    
    return image

def predict_with_model(csv_file, model_path):
    print(f"Loading model from: {model_path}")
    print(f"Reading CSV file from: {csv_file}")
    df = pd.read_csv(csv_file)
    print("CSV Data Loaded:", df.head())
    side_1_df = df[['X', 'Y', 'side1']].rename(columns={'side1': 'THK'})
    side_2_df = df[['X', 'Y', 'side2']].rename(columns={'side2': 'THK'})
    
    side_1_image = process_csv(side_1_df)
    side_2_image = process_csv(side_2_df)
    print("Processed images for side 1 and side 2")
    model = load_model(model_path)
    print("Model loaded successfully")
    side_1_input_image = side_1_image.reshape(1, side_1_image.shape[0], side_1_image.shape[1], 1)
    side_2_input_image = side_2_image.reshape(1, side_2_image.shape[0], side_2_image.shape[1], 1)
    
    side_1_predictions = model.predict(side_1_input_image)
    side_2_predictions = model.predict(side_2_input_image)
    print("Predictions made for both sides")
    side_1_predictions[0][0] *= -1
    side_1_predictions[0][1] *= -1
    side_1_predictions[0][2] *= 1
    
    side_2_predictions[0][0] *= -1
    side_2_predictions[0][1] *= -1
    side_2_predictions[0][2] *= -1
    
    print("Predictions adjusted:", side_1_predictions, side_2_predictions)
    
    return side_1_predictions[0], side_2_predictions[0]



class PredictionApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Kodiak U% Tuning Prediction App")
        self.geometry("600x400")
        
        self.model_path = "site1_model_iteration_8.h5"  # Replace with your model path
        self.csv_file = None
        self.result_df = None
        
        
        
        
        # 使用 grid 將整個視窗分為 5x5
        for row in range(5):
            self.grid_rowconfigure(row, weight=1, minsize=80)  # 每行的最小高度
        for col in range(5):
            self.grid_columnconfigure(col, weight=1, minsize=120)
        
        self.drop_area = tk.Label(self, text="Drag and Drop CSV File Here", bg="lightgray", width=30, height=10)
        self.drop_area.grid(row= 0 , column = 0 , columnspan = 2 , pady=20)
        
        self.drop_area.drop_target_register(DND_FILES)
        self.drop_area.dnd_bind('<<Drop>>', self.on_file_drop)

        
        self.k_frame = tk.Frame(self)
        self.k_frame.grid(row = 1,column = 0,rowspan = 3,columnspan = 5, padx= 10 , pady = 10, sticky = "nsew")

        # 创建两个 Entry 小部件用于输入 K 值
        self.k_value_1_label = tk.Label(self.k_frame, text="K value 1:")
        self.k_value_1_label.grid(row = 0, column = 0 , padx = 10, pady=10,sticky = "e")
        self.k_value_1_entry = tk.Entry(self.k_frame)
        self.k_value_1_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        # K Value 2 Label 和 Entry (輸入框)
        self.k_value_2_label = tk.Label(self.k_frame, text="K Value 2:")
        self.k_value_2_label.grid(row=0, column=2, padx=10, pady=10, sticky="e")
        self.k_value_2_entry = tk.Entry(self.k_frame)
        self.k_value_2_entry.grid(row=0, column=3, padx=10, pady=10, sticky="w")
        
        
        self.THK_1_label = tk.Label(self.k_frame, text="THK 1:")
        self.THK_1_label.grid(row = 1, column = 0 , padx = 10, pady=10,sticky = "e")
        self.THK_1_entry = tk.Entry(self.k_frame)
        self.THK_1_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        # K Value 2 Label 和 Entry (輸入框)
        self.THK_2_label = tk.Label(self.k_frame, text="THK 2:")
        self.THK_2_label.grid(row=1, column=2, padx=10, pady=10, sticky="e")
        self.THK_2_entry = tk.Entry(self.k_frame)
        self.THK_2_entry.grid(row=1, column=3, padx=10, pady=10, sticky="w")
        
        
        self.Dep_1_label = tk.Label(self.k_frame, text="Dep 1 now:")
        self.Dep_1_label.grid(row = 2, column = 0 , padx = 10, pady=10,sticky = "e")
        self.Dep_1_entry = tk.Entry(self.k_frame)
        self.Dep_1_entry.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        # K Value 2 Label 和 Entry (輸入框)
        self.Dep_2_label = tk.Label(self.k_frame, text="Dep 2 now:")
        self.Dep_2_label.grid(row=2, column=2, padx=10, pady=10, sticky="e")
        self.Dep_2_entry = tk.Entry(self.k_frame)
        self.Dep_2_entry.grid(row=2, column=3, padx=10, pady=10, sticky="w")
        
        
                


        

        
        # Plot 按鈕
        self.plot_button = tk.Button(self, text="Plot", command=self.show_U_map)
        self.plot_button.grid(row=3, column=0, columnspan=2, pady=20)
        
        # 預測按鈕
        self.predict_button = tk.Button(self, text="Start Prediction", command=self.start_prediction)
        self.predict_button.grid(row=4, column=0, columnspan=2, pady=20)
        
        
        self.tree = ttk.Treeview(self, columns=('OZO', '左後/左前', '右前/右後', 'PV_offset'), show='headings', height=5)
        self.tree.grid(pady=20)

        # Define column headings
        self.tree.heading('OZO', text='OZO', anchor='center')
        self.tree.heading('左後/左前', text='左後/左前', anchor='center')
        self.tree.heading('右前/右後', text='右前/右後', anchor='center')
        self.tree.heading('PV_offset', text='PV_offset', anchor='center')

        # Center alignment for the columns
        self.tree.column('OZO', anchor='center', width=100)
        self.tree.column('左後/左前', anchor='center', width=100)
        self.tree.column('右前/右後', anchor='center', width=100)
        self.tree.column('PV_offset', anchor='center', width=100)
        
        #self.save_button = tk.Button(self, text="Save to CSV", command=self.save_to_csv)
        #self.save_button.pack(pady=20)
        # Canvas 顯示區域
        self.canvas = tk.Canvas(self, width=500, height=500, bg="white")
        self.canvas.grid(row=3, column=2, columnspan=3, rowspan=2, sticky="nsew", padx=10, pady=10)

        self.result_df = None 
        
          # 假设预测结果会存储在 result_df 中  
    def on_file_drop(self, event):
        self.csv_file = event.data.strip('{}')
        self.drop_area.config(text=os.path.basename(self.csv_file))
        
    def show_U_map(self):
        if not self.csv_file:
            messagebox.showerror("Error", "No CSV file selected!")
            return
        try:
            df_1 = process_data_for_plot("side1",self.csv_file)
            df_2 = process_data_for_plot("side2",self.csv_file)
            
            grid_x, grid_y = np.mgrid[
                -max(df_1['radius']):max(df_1['radius']):200j,  # 200j 表示生成 200x200 的网格
                -max(df_1['radius']):max(df_1['radius']):200j
                ]
            
            grid_z_thk_1 = griddata(
                (df_1['X'], df_1['Y']),  # 原始數據點位置
                df_1["side1"],           # 對應的值
                (grid_x, grid_y),          # 插值網格點
                method='linear'             # 插值方法 ('linear', 'nearest', 'cubic')
                )
            grid_z_radial_1 = griddata(
                (df_1['X'], df_1['Y']),  # 原始數據點位置
                df_1['Radial'],           # 對應的值
                (grid_x, grid_y),          # 插值網格點
                method='linear'             # 插值方法 ('linear', 'nearest', 'cubic')
                )
            grid_z_planar_1 = griddata(
                (df_1['X'], df_1['Y']),  # 原始數據點位置
                df_1['Planar'],          # 對應的值 (Planar)
                (grid_x, grid_y),          # 插值網格點
                method='linear'             # 插值方法 ('linear', 'nearest', 'cubic')
                )

            grid_z_residual_1 = griddata(
                (df_1['X'], df_1['Y']),  # 原始數據點位置
                df_1['Residual'],          # 對應的值 (Resiual)
                (grid_x, grid_y),          # 插值網格點
                method='linear'             # 插值方法 ('linear', 'nearest', 'cubic')
                )
            
            
            fig, axs = plt.subplots(2, 4, figsize=(32, 8))
            contour_thk = axs[0,0].contourf(grid_x, grid_y, grid_z_thk_1, levels=20, cmap='jet', vmin = 2450, vmax = 2600)
            fig.colorbar(contour_thk, ax=axs[0,0], label='THK value')
            
            circle_thk = plt.Circle((0, 0), max(df_1['radius']), color='black', fill=False, lw=2)
            axs[0,0].add_artist(circle_thk)
            
            axs[0,0].set_xlim(-max(df_1['radius'])-20, max(df_1['radius'])+20)
            axs[0,0].set_ylim(-max(df_1['radius'])-20, max(df_1['radius'])+20)

            axs[0,0].set_aspect('equal', adjustable='box')

            axs[0,0].set_xlabel('X Position')
            axs[0,0].set_ylabel('Y Position')
            axs[0,0].set_title("Side1 " + 'THK',fontsize = 16)
            
            
         
            contour_radial = axs[0,1].contourf(grid_x, grid_y, grid_z_radial_1, levels=20, cmap='jet', vmin = -50, vmax = 50)
            fig.colorbar(contour_radial, ax=axs[0,1], label='Radial value')


            circle_radial = plt.Circle((0, 0), max(df_1['radius']), color='black', fill=False, lw=2)
            axs[0,1].add_artist(circle_radial)


            axs[0,1].set_xlim(-max(df_1['radius'])-20, max(df_1['radius'])+20)
            axs[0,1].set_ylim(-max(df_1['radius'])-20, max(df_1['radius'])+20)


            axs[0,1].set_aspect('equal', adjustable='box')

            axs[0,1].set_xlabel('X Position')
            axs[0,1].set_ylabel('Y Position')
            axs[0,1].set_title("Side1 " + 'Radial',fontsize = 16)

            contour_planar = axs[0,2].contourf(grid_x, grid_y, grid_z_planar_1, levels=20, cmap='jet', vmin = -50, vmax = 50)
            fig.colorbar(contour_planar, ax=axs[0,2], label='Planar value')


            circle_planar = plt.Circle((0, 0), max(df_1['radius']), color='black', fill=False, lw=2)
            axs[0,2].add_artist(circle_planar)


            axs[0,2].set_xlim(-max(df_1['radius'])-20, max(df_1['radius'])+20)
            axs[0,2].set_ylim(-max(df_1['radius'])-20, max(df_1['radius'])+20)


            axs[0,2].set_aspect('equal', adjustable='box')
    
            axs[0,2].set_xlabel('X Position')
            axs[0,2].set_ylabel('Y Position')
            axs[0,2].set_title("Side1 " + 'Planar', fontsize=16)
            

            contour_residual = axs[0,3].contourf(grid_x, grid_y, grid_z_residual_1, levels=20, cmap='jet',vmin = 2450, vmax = 2600)
            fig.colorbar(contour_residual, ax=axs[0,3], label='Residual value')


            circle_residual = plt.Circle((0, 0), max(df_1['radius']), color='black', fill=False, lw=2)
            axs[0,3].add_artist(circle_residual)

   
            axs[0,3].set_xlim(-max(df_1['radius'])-20, max(df_1['radius'])+20)
            axs[0,3].set_ylim(-max(df_1['radius'])-20, max(df_1['radius'])+20)
    

            axs[0,3].set_aspect('equal', adjustable='box')
            
            axs[0,3].set_xlabel('X Position')
            axs[0,3].set_ylabel('Y Position')
            axs[0,3].set_title("Side1 "  + 'Residual', fontsize=16)            
            
            grid_z_thk_2 = griddata(
                (df_2['X'], df_2['Y']),  # 原始數據點位置
                df_2["side2"],           # 對應的值
                (grid_x, grid_y),          # 插值網格點
                method='linear'             # 插值方法 ('linear', 'nearest', 'cubic')
                )
            grid_z_radial_2 = griddata(
                (df_2['X'], df_2['Y']),  # 原始數據點位置
                df_2['Radial'],           # 對應的值
                (grid_x, grid_y),          # 插值網格點
                method='linear'             # 插值方法 ('linear', 'nearest', 'cubic')
                )
            grid_z_planar_2 = griddata(
                (df_2['X'], df_2['Y']),  # 原始數據點位置
                df_2['Planar'],          # 對應的值 (Planar)
                (grid_x, grid_y),          # 插值網格點
                method='linear'             # 插值方法 ('linear', 'nearest', 'cubic')
                )

            grid_z_residual_2 = griddata(
                (df_2['X'], df_2['Y']),  # 原始數據點位置
                df_2['Residual'],          # 對應的值 (Resiual)
                (grid_x, grid_y),          # 插值網格點
                method='linear'             # 插值方法 ('linear', 'nearest', 'cubic')
                )           
            contour_thk = axs[1,0].contourf(grid_x, grid_y, grid_z_thk_2, levels=20, cmap='jet', vmin = 2450, vmax = 2600)
            fig.colorbar(contour_thk, ax=axs[1,0], label='THK value')
            
            circle_thk = plt.Circle((0, 0), max(df_2['radius']), color='black', fill=False, lw=2)
            axs[1,0].add_artist(circle_thk)
            
            axs[1,0].set_xlim(-max(df_2['radius'])-20, max(df_2['radius'])+20)
            axs[1,0].set_ylim(-max(df_2['radius'])-20, max(df_2['radius'])+20)

            axs[1,0].set_aspect('equal', adjustable='box')

            axs[1,0].set_xlabel('X Position')
            axs[1,0].set_ylabel('Y Position')
            axs[1,0].set_title("Side2 " + 'THK',fontsize = 16)
            
            
         
            contour_radial = axs[1,1].contourf(grid_x, grid_y, grid_z_radial_2, levels=20, cmap='jet', vmin = -50, vmax = 50)
            fig.colorbar(contour_radial, ax=axs[1,1], label='Radial value')


            circle_radial = plt.Circle((0, 0), max(df_2['radius']), color='black', fill=False, lw=2)
            axs[1,1].add_artist(circle_radial)


            axs[1,1].set_xlim(-max(df_2['radius'])-20, max(df_2['radius'])+20)
            axs[1,1].set_ylim(-max(df_2['radius'])-20, max(df_2['radius'])+20)


            axs[1,1].set_aspect('equal', adjustable='box')

            axs[1,1].set_xlabel('X Position')
            axs[1,1].set_ylabel('Y Position')
            axs[1,1].set_title("Side2 " + 'Radial',fontsize = 16)

            contour_planar = axs[1,2].contourf(grid_x, grid_y, grid_z_planar_2, levels=20, cmap='jet', vmin = -50, vmax = 50)
            fig.colorbar(contour_planar, ax=axs[1,2], label='Planar value')


            circle_planar = plt.Circle((0, 0), max(df_2['radius']), color='black', fill=False, lw=2)
            axs[1,2].add_artist(circle_planar)


            axs[1,2].set_xlim(-max(df_2['radius'])-20, max(df_2['radius'])+20)
            axs[1,2].set_ylim(-max(df_2['radius'])-20, max(df_2['radius'])+20)


            axs[1,2].set_aspect('equal', adjustable='box')
    
            axs[1,2].set_xlabel('X Position')
            axs[1,2].set_ylabel('Y Position')
            axs[1,2].set_title("Side2 " + 'Planar', fontsize=16)
            

            contour_residual = axs[1,3].contourf(grid_x, grid_y, grid_z_residual_2, levels=20, cmap='jet',vmin = 2450, vmax = 2600)
            fig.colorbar(contour_residual, ax=axs[1,3], label='Residual value')


            circle_residual = plt.Circle((0, 0), max(df_2['radius']), color='black', fill=False, lw=2)
            axs[1,3].add_artist(circle_residual)

   
            axs[1,3].set_xlim(-max(df_2['radius'])-20, max(df_2['radius'])+20)
            axs[1,3].set_ylim(-max(df_2['radius'])-20, max(df_2['radius'])+20)
    

            axs[1,3].set_aspect('equal', adjustable='box')
            
            axs[1,3].set_xlabel('X Position')
            axs[1,3].set_ylabel('Y Position')
            axs[1,3].set_title("Side2 "  + 'Residual', fontsize=16)  
            plt.tight_layout()
            plt.show()
            
            
            #plt.figure(figsize=(6, 4))
            #plt.scatter(df['X'], df['Y'])
            #plt.xlabel('X')
            #plt.ylabel('Y')
            #plt.title('Scatter Plot of X vs Y')
            
            #self.plot_canvas = FigureCanvasTkAgg(plt.gcf(), master=self)
            #self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            #self.plot_canvas.draw()
                        
        except Exception as e:
            print("An error occurred while plotting:", str(e))
            messagebox.showerror("Error", f"An unexpected error occurred while plotting: {str(e)}")

    def start_prediction(self):
        if not self.csv_file:
            messagebox.showerror("Error", "No CSV file selected!")
            return

        try:

            print(f"Starting prediction for: {self.csv_file}")
            side_1_result, side_2_result = predict_with_model(self.csv_file, self.model_path)
            print("Prediction results:", side_1_result, side_2_result)
            ##### deal with PV_offset tuning
            OZO_modify_1 = math.floor(round(side_1_result[0])/2)
            OZO_modify_2 = math.floor(round(side_2_result[0])/2)
            print(OZO_modify_1)
            print(OZO_modify_2)
            k_value_1 = math.floor(-(0.662 - float(float(self.k_value_1_entry.get())))/0.0015) + OZO_modify_1
            k_value_2 = math.floor(-(0.662 - float(float(self.k_value_2_entry.get())))/0.0015) + OZO_modify_2
            print(type(k_value_1))
            print(type(k_value_2))
            #### deal with SDT tuning
            dep_rate_1 = float(float(self.THK_1_entry.get()))/float(self.Dep_1_entry.get())
            dep_rate_2 = float(float(self.THK_2_entry.get()))/float(self.Dep_2_entry.get())
            print(type(dep_rate_1))
            print(type(dep_rate_2))
            def check_value(val):
                
                return "Keep" if -5<=val<= 5 else val

            self.result_df = pd.DataFrame({
                'Side': ['Side 1', 'Side 2'],
                'OZO': [round(side_1_result[0]), round(side_2_result[0])],
                '左後/左前': [check_value(side_1_result[2]), check_value(side_2_result[1])],
                '右前/右後': [check_value(side_1_result[1]), check_value(side_2_result[2])],
                "PV_offset":[k_value_1,k_value_2],
                "New SDT":[round(2500/dep_rate_1,1),round(2500/dep_rate_2,1)]
            })
            # Clear previous results from the Treeview
 
            print(self.result_df)
            print("Result DataFrame created successfully:", self.result_df)
            
            
    
        except Exception as e:
            print("An error occurred during prediction:", str(e))
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")
        self.display_results_on_canvas()
    def display_results_on_canvas(self):
        if self.result_df is not None:
            try:
                print("Displaying result DataFrame on Canvas...")
                print("DataFrame content:")
                print(self.result_df)
            
            # 清除之前的内容
                self.canvas.delete("all")
              
           
            # 设置列宽度
                col_width = 60  # 你可以根据需要调整列宽
                row_height = 20  # 行高
                start_x = 30     # 起始X坐标
                start_y = 20     # 起始Y坐标

            # 绘制列名
                for col_idx, col_name in enumerate(self.result_df.columns):
                    x = start_x + col_idx * col_width
                    self.canvas.create_text(x, start_y, text=col_name, font=("Arial", 10), anchor="center")

            # 绘制每行数据
                for row_idx, row in self.result_df.iterrows():
                    for col_idx, value in enumerate(row):
                        x = start_x + col_idx * col_width
                        y = start_y + (row_idx + 1) * row_height
                        #formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                        self.canvas.create_text(x, y, text=f"{value:.3f}" if isinstance(value, float) else str(value), font=("Arial", 10), anchor="center")
            
                print("Results displayed on Canvas.")
            except Exception as e:
                    print(f"An error occurred while displaying results on Canvas: {str(e)}")
                    messagebox.showerror("Error", f"An error occurred while displaying results: {str(e)}")
        else:
            print("No data to display: result_df is None")
            messagebox.showwarning("No Data", "No prediction results to display! Please run a prediction first.")


if __name__ == "__main__":
    app = PredictionApp()
    app.mainloop()
