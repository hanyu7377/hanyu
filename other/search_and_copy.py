
import pandas as pd
import numpy as np
import openpyxl
import shutil
from openpyxl.styles import Font
import os
from datetime import datetime
count = 0

# 照片搜索函数


def search_photos(excel_path, photo_folder):
    df = pd.read_csv(excel_path, encoding='latin1')  # 讀取 CSV 檔案
    photo_folders = [folder for folder in os.listdir(
        photo_folder) if os.path.isdir(os.path.join(photo_folder, folder))]

    for index, row in df.iterrows():  # 逐行讀取 Excel 資料
        camera_id = row['Camera']  # 相機ID
        photo_name = row['File name']  # 照片檔案名稱
        photo_datetime = None
        try:
            photo_datetime = datetime.strptime(
                row['Date & Time'], '%Y/%m/%d %H:%M')
        except ValueError:
            try:
                photo_datetime = datetime.strptime(
                    row['Date & Time'], '%m/%d/%Y %H:%M')
            except ValueError:
                try:
                    photo_datetime = datetime.strptime(
                        row['Date & Time'], '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    print(f'無法解析日期時間：{row["Date & Time"]}')
                    continue

        target_folder = None  # 目標資料夾
        for folder in photo_folders:
            if folder.startswith(camera_id):  # 檢查相機ID是否匹配
                date_strs = folder.rsplit('-', 2)[1:]
                folder_start_date = datetime.strptime(date_strs[0], '%Y%m%d')
                folder_end_date = datetime.strptime(date_strs[1], '%Y%m%d')
                if folder_start_date <= photo_datetime <= folder_end_date:
                    target_folder = os.path.join(photo_folder, folder)
                    break

        if target_folder:
            photo_path = os.path.join(target_folder, photo_name)  # 照片完整路徑
            if os.path.exists(photo_path):
                print(f'找到照片：{photo_path}')
                target_folder_path = os.path.join('D:', target_folder)
                os.makedirs(target_folder_path, exist_ok=True)
                shutil.copy(photo_path, target_folder_path)
                print(f'已複製照片至：{target_folder_path}')
            else:
                print(f'找不到照片：{photo_path}')
        else:
            print(f'找不到對應的資料夾：{photo_datetime}')


# 主程式碼
root_folder = '.'  # 主程式所在的資料夾
excel_files = [file for file in os.listdir(
    root_folder) if file.endswith('.csv')]
print(excel_files)
photo_folders = [folder for folder in os.listdir(
    root_folder) if os.path.isdir(folder)]
del photo_folders[0]
del photo_folders[-2]
print(photo_folders)
count = 0
for excel_file, photo_folder in zip(excel_files, photo_folders):
    count += 1
    print(excel_files)
    print(photo_folders)
    excel_path = os.path.join(root_folder, excel_file)
    photo_folder_path = os.path.join(root_folder, photo_folder)

    target_folder_root = 'D:'  # D盘根目录
    target_folder_path = os.path.join(target_folder_root, photo_folder)
    os.makedirs(target_folder_path, exist_ok=True)  # 创建目标文件夹

    search_photos(excel_path, photo_folder_path)
    print(count)

print(count)
