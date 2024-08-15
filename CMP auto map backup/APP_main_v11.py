# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:19:48 2024

@author: HANYUHSIAO
"""


import os
import tkinter as tk
from tkinter import  messagebox, ttk
import pandas as pd
import math
from tkinterdnd2 import DND_FILES, TkinterDnD
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg

print("Starting the application...")
tkdnd_path = r'C:\ProgramData\anaconda3\tcl\tkdnd2.8'
os.environ['TKDND_LIBRARY'] = tkdnd_path


root = TkinterDnD.Tk()
root.title("Data Analysis Application")

# 保存Y轴数据、标签和上下限
saved_y_data = None
saved_y_label = None
saved_upper_limit = None
saved_lower_limit = None
class plot_setting():
    def add_text(self,x,y,string):
        plt.text(x, y, string , ha='center', va='bottom')
plot_setting = plot_setting()  

def analyze_OCD():
    # 创建OCD分析窗口
    global ocd_window
    ocd_window = tk.Toplevel(root)
    ocd_window.title("OCD Analysis")
    ocd_window.geometry("600x600")

    # 创建网格布局
    for i in range(0,8,1):
        ocd_window.grid_rowconfigure(i, weight=1)
    for j in range(0,6,1):
        ocd_window.grid_columnconfigure(j, weight=1)


    ######################### 拖放区域
    drop_area = tk.Label(ocd_window, text="Drop Area", bg="lightgray", width=30, height=6)
    drop_area.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

    label = tk.Label(ocd_window, text="Drag and Drop Excel File Here", font=("Arial", 10))
    label.grid(row=0, column=0, columnspan=2, padx=10, pady=5)

    drop_area.drop_target_register(DND_FILES)
    drop_area.dnd_bind('<<Drop>>', lambda event: drop(event, ocd_window))
    ######################################
    
    
    ########################################### slot选择槽位
    slot_label = tk.Label(ocd_window, text="Select Slots", font=("Arial", 14))
    slot_label.grid(row=1, column=0, padx=10, pady=10, sticky='w')

    global slot_listbox
    slot_listbox = tk.Listbox(ocd_window, selectmode=tk.MULTIPLE, width=5, height=3)
    slot_listbox.grid(row=2, column=0, padx=10, pady=10, sticky='nsew')
    slot_listbox.config(width = 10 ,height = 5)
    ########################################################
    
    
    #######################select slot decision part
    selection_button_frame = tk.Frame(ocd_window)
    selection_button_frame.grid(row=3, column=0, rowspan=2, columnspan=1, padx=10, pady=(0,0), sticky='nsew')
    
    
    select_all_button = tk.Button(selection_button_frame, text="Select All", command=lambda: select_all_slots(slot_listbox))
    select_all_button.grid(row=0, column=0, padx=(10,0), pady=5, sticky='ew')
    select_all_button.config(width=20, height=1)

    deselect_all_button = tk.Button(selection_button_frame, text="Deselect All", command=lambda: deselect_all_slots(slot_listbox))
    deselect_all_button.grid(row=1, column=0, padx=(10,0), pady=5, sticky='ew')
    deselect_all_button.config(width=20, height=1)
    ################################################################
    
    
    ##########################################標題輸入位置
    title_frame = tk.Frame(ocd_window)
    title_frame.grid(row=0, column=3, rowspan=3, columnspan=2, padx=(10,50), pady=100, sticky='nsew')

    title_label = tk.Label(title_frame, text="Plot Title", font=("Arial", 10))
    title_label.grid(row=0, column=1, padx=10, pady=5, sticky='w')

    global title_entry
    title_entry = tk.Entry(title_frame)
    title_entry.grid(row=0, column=2, padx=100, pady=5, columnspan=2)
    #########################################
    
    ################################ title , key, Y轴数据和标签选择
    selection_frame = tk.Frame(ocd_window)
    selection_frame.grid(row=1, column=3, rowspan=4, columnspan=2, padx=10, pady=5, sticky='nsew')

    key_label =tk.Label(selection_frame, text = "Whick key ?", font = ("Arial", 10))
    key_label.grid(row=0, column=0, padx=(10,0), pady=5, sticky='w')
    global key_combobox
    key_combobox = ttk.Combobox(selection_frame, state='readonly')
    key_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
    key_combobox.bind("<<ComboboxSelected>>", on_key_selected)
    
    
    
    
    
    y_label = tk.Label(selection_frame, text="Select Y Axis Data (film)", font=("Arial", 10))
    y_label.grid(row=1, column=0, padx=10, pady=5, sticky='w')
    y_options = ["T1", "T2", "T3", "T4"]
    global y_combobox
    y_combobox = ttk.Combobox(selection_frame, values=y_options)
    y_combobox.grid(row=1, column=1, padx=10, pady=5)

    label_label = tk.Label(selection_frame, text="Select Y Label", font=("Arial", 10))
    label_label.grid(row=2, column=0, padx=10, pady=5, sticky='w')
    label_options = ["Bias (Å)", "THK (Å)"]
    global label_combobox
    label_combobox = ttk.Combobox(selection_frame, values=label_options)
    label_combobox.grid(row=2, column=1, padx=10, pady=5)
    #######################################
    #############作圖setting 
    Line_frame = tk.Frame(ocd_window)
    Line_frame.grid(row=3, column=3, columnspan=2, padx=10, pady=5, sticky='w')

    global upper_spec_entry, lower_spec_entry, upper_limit_entry, lower_limit_entry

    upper_limit_label = tk.Label(Line_frame, text="Upper Limit", font=("Arial", 14))
    upper_limit_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
    upper_limit_entry = tk.Entry(Line_frame)
    upper_limit_entry.grid(row=0, column=1, padx=5, pady=5)
    
    upper_spec_label = tk.Label(Line_frame, text="Upper Spec", font=("Arial", 14))
    upper_spec_label.grid(row=1, column=0, padx=5, pady=5, sticky='w')
    upper_spec_entry = tk.Entry(Line_frame)
    upper_spec_entry.grid(row=1, column=1, padx=5, pady=5)
    
    

    lower_spec_label = tk.Label(Line_frame, text="Lower Spec", font=("Arial", 14))
    lower_spec_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')
    lower_spec_entry = tk.Entry(Line_frame)
    lower_spec_entry.grid(row=2, column=1, padx=5, pady=5)

    lower_limit_label = tk.Label(Line_frame, text="Lower Limit", font=("Arial", 14))
    lower_limit_label.grid(row=3, column=0, padx=5, pady=5, sticky='w')
    lower_limit_entry = tk.Entry(Line_frame)
    lower_limit_entry.grid(row=3, column=1, padx=5, pady=5)

    # 图表标题输入框


    # 按钮
    button_frame = tk.Frame(ocd_window)
    button_frame.grid(row=6, column=1, columnspan=3, padx=10, pady=10)

    plot_button = tk.Button(button_frame, text="Plot", command=lambda: plot_data_ocd())
    plot_button.pack(side=tk.LEFT, padx=5)

    back_button = tk.Button(button_frame, text="Back", command=ocd_window.destroy)
    back_button.pack(side=tk.RIGHT, padx=5)

def drop(event, window, is_nova_cpp05 = False):
    print("File dropped.")
    file_path = event.data.strip('{}')
    print(f"Dropped file path: {file_path}")
    if file_path:
        process_file(file_path, window , is_nova_cpp05)




def on_key_selected(event):
    selected_key = key_combobox.get()
    selected_key = float(selected_key)
    print("value is: ", selected_key , "and type is: ", type(selected_key))
    filtered_slot = df[df["Test ID"] == selected_key]["Slot #"].unique().tolist()
    print(filtered_slot)
    if selected_key:
        # Filter DataFrame to only include data for the selected key
        filtered_df = df[df["Test ID"] == selected_key]
        
        # Ensure 'Slot #' column exists in the filtered DataFrame
        if "Slot #" in filtered_df.columns:
            # Get unique slots associated with the selected key
            slots = filtered_df["Slot #"].unique()
            
            # Clear the slot_listbox and populate it with the filtered slots
            slot_listbox.delete(0, tk.END)
            for slot in sorted(slots):  # Sort and insert slots into listbox
                slot_listbox.insert(tk.END, int(slot))
        else:
            messagebox.showerror("Error", "'Slot #' column not found in the filtered data.")


def process_file(file_path, window, is_nova_cpp05=False):
    try:
        global df
        df = pd.read_csv(file_path)

        
        df = df.loc[:, ~df.columns.duplicated()]
        # Determine the appropriate column name for slots
        if is_nova_cpp05:
            slot_column = "Slot"
        else:
            slot_column = "Slot #"
        
        if slot_column not in df.columns:
            messagebox.showerror("Error", f"The file does not contain '{slot_column}' column")
            return
        df[slot_column] = df[slot_column].astype(str).str.strip()
        
        # Filter out non-numeric slots
        df = df[df[slot_column].str.isnumeric()]
        
        # Ensure the slot column is numeric
        df[slot_column] = pd.to_numeric(df[slot_column], errors='coerce')
        
        # Get unique slots
       
    
        # Remove duplicate column names if required
        if is_nova_cpp05:
            
            print(f"DataFrame columns after removing duplicates: {df.columns}")        
            slots_combobox['values'] = sorted(df[slot_column].unique())
            datetime_combobox['values'] = []      
            print(type(df["DateTime"]))
            print(type(df["DateTime"][3]))
            print(type(df["Slot"]))
            print(type(df["Slot"][3]))
            
            print(type(df["MpointX"][3]))
            df['MpointX'] = pd.to_numeric(df['MpointX'], errors='coerce')
            df['MpointY'] = pd.to_numeric(df['MpointY'], errors='coerce')
            df['OX_HEIGHT'] = pd.to_numeric(df['OX_HEIGHT'], errors='coerce')
            print(type(df["MpointX"][3]))
            print(type(df["MpointY"][3]))
            print(type(df["OX_HEIGHT"][3]))
            
        # Clear and populate the listbox with numeric slots
        if not is_nova_cpp05:
            # 为 key_combobox 设置可选值
            key_combobox['values'] = sorted(df["Test ID"].unique())
            # 预先清空 slot_listbox
            slot_listbox.delete(0, tk.END)
        print(df)  
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process file: {e}")
        
        




def select_all_slots(listbox):
    listbox.select_set(0, tk.END)

def deselect_all_slots(listbox):
    listbox.select_clear(0, tk.END)

def plot_data_ocd():
    key = float(key_combobox.get())
    print("value is: ", key , "and type is: ", type(key))
   
    #print("value is: ", selected_key , "and type is: ", type(selected_key))
    #filtered_slot = df[df["Test ID"] == selected_key]["Slot #"].unique().tolist()
    #print(filtered_slot)
    
    selected_slots = [slot_listbox.get(i) for i in slot_listbox.curselection()]
    
    print(f"Selected slots: {selected_slots}")
    
    if not selected_slots:
        messagebox.showerror("Error", "No slots selected")
        return
    filtered_df = df[df["Test ID"] == key]
    filtered_df = filtered_df[filtered_df["Slot #"].isin(selected_slots)]
   
    print(filtered_df[["Slot #", "Test ID"]])
    
    if filtered_df.empty:
        messagebox.showerror("Error", "No data for selected slots")
        return
    
    y_data = y_combobox.get()
    y_label = label_combobox.get()
    plot_title = title_entry.get()
    try:
        
        upper_limit = float(upper_limit_entry.get())
        upper_spec = float(upper_spec_entry.get())
        lower_spec = float(lower_spec_entry.get())
        lower_limit = float(lower_limit_entry.get())
        
    except ValueError:
        messagebox.showerror("Error", "Upper and Lower limits must be numbers")
        return

    global saved_y_data, saved_y_label, saved_upper_limit, saved_lower_limit, save_upper_spec, save_lower_spec
    saved_y_data = y_data
    saved_y_label = y_label
    saved_upper_limit = upper_limit
    saved_lower_limit = lower_limit
    save_upper_spec = upper_spec
    save_lower_spec = lower_spec

    target = (upper_spec + lower_spec) / 2
    
    zone_mark_position = saved_lower_limit + (saved_upper_limit -saved_lower_limit) * 0.02
    filtered_df['Radius (mm)'] = (filtered_df['Wafer X'] / 1000) ** 2 + (filtered_df['Wafer Y'] / 1000) ** 2
    filtered_df['Radius (mm)'] = filtered_df['Radius (mm)'].apply(math.sqrt)

    print(f"Filtered DataFrame after processing:\n{filtered_df}")

    plt.figure()

    for slot in filtered_df['Slot #'].unique():
        slot_data = filtered_df[filtered_df['Slot #'] == slot]
        print(f"Plotting Slot {slot}: {slot_data.head(20)}")
        plt.scatter(slot_data['Radius (mm)'], slot_data[y_data], label=f'Slot {slot}')
    if upper_spec and lower_spec:
        
        plt.axhline(y=upper_spec, color='r', linestyle='--', label='Upper Limit')
        plt.axhline(y=lower_spec, color='r', linestyle='--', label='Lower Limit')
        
    plt.axhline(y=target, color='r', linestyle='-', label='Target')
    plot_setting.add_text(110, zone_mark_position, "Z4")
    plot_setting.add_text(130, zone_mark_position, "Z3")
    plot_setting.add_text(140, zone_mark_position, "Z2")
    plot_setting.add_text(146, zone_mark_position, "Z1")
    plt.xlabel("Radius (mm)", fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.title(plot_title if plot_title else f"Scatter Plot of {y_data} vs Radius", fontsize=24)
    plt.ylim([lower_limit, upper_limit])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(range(0, 151, 15))
    plt.show()
    ocd_window.deiconify() 

def analyze_siview():
    messagebox.showinfo("Info", "siview analysis selected")
########################################################################
def analyze_Nova_CPP05():
    global nova_cpp05_window
    nova_cpp05_window = tk.Toplevel(root)
    nova_cpp05_window.title("Nova_cpp05 Analysis")
    nova_cpp05_window.geometry("700x800")
    
    for i in range(0, 10, 1):
        nova_cpp05_window.grid_rowconfigure(i, weight=1)
    for j in range(0, 6, 1):
        nova_cpp05_window.grid_columnconfigure(j, weight=1)

    drop_area = tk.Label(nova_cpp05_window, text="Drop Area", bg="lightgray", width=30, height=6)
    drop_area.grid(row=0, column=0, columnspan=2, padx=10, pady=5)

    label = tk.Label(nova_cpp05_window, text="Drag and Drop Excel File Here", font=("Arial", 10))
    label.grid(row=0, column=0, columnspan=2, padx=10, pady=5)

    drop_area.drop_target_register(DND_FILES)
    drop_area.dnd_bind('<<Drop>>', lambda event: drop(event, nova_cpp05_window, is_nova_cpp05=True))
    
    # Slots label and combobox
    slots_label = tk.Label(nova_cpp05_window, text='Slots:', font=("Arial", 14))
    slots_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
    global slots_combobox
    slots_combobox = ttk.Combobox(nova_cpp05_window, state='readonly')
    slots_combobox.grid(row=1, column=1, padx=5, pady=5, sticky="w")
    slots_combobox.bind("<<ComboboxSelected>>", update_datetime_combobox)
    
    # DateTime label and combobox
    datetime_label = tk.Label(nova_cpp05_window, text='DateTime:', font=("Arial", 14))
    datetime_label.grid(row=2, column=0, padx=(2,2), pady=1, sticky="e")
    global datetime_combobox
    datetime_combobox = ttk.Combobox(nova_cpp05_window, state='readonly')
    datetime_combobox.grid(row=2, column=1, padx=5, pady=5, sticky="w")
    
    # Bucket feature
    bucket_label = tk.Label(nova_cpp05_window, text="Bucket:", font=("Arial", 14))
    bucket_label.grid(row=3, column=0, padx=(3, 0), pady=1, sticky='e')

    global bucket_listbox
    bucket_listbox = tk.Listbox(nova_cpp05_window,selectmode=tk.MULTIPLE, width=30, height=6)
    bucket_listbox.grid(row=3, column=1, padx=(0, 10), pady=(0,0), sticky='w')

    title_label = tk.Label(nova_cpp05_window, text="Plot Title", font=("Arial", 14))
    title_label.grid(row=1, column=2, padx=5, pady=1, sticky='w')

    global title_entry
    title_entry = tk.Entry(nova_cpp05_window)
    title_entry.grid(row=1, column=3, padx=5, pady=5, columnspan=1)


    label_label = tk.Label(nova_cpp05_window, text="Select Y Label", font=("Arial", 14))
    label_label.grid(row=2, column=2, padx=10, pady=5, sticky='w')
    label_options = ["Bias (Å)", "THK (Å)"]
    global label_combobox
    label_combobox = ttk.Combobox(nova_cpp05_window, values=label_options)
    label_combobox.grid(row=2, column=3, padx=10, pady=5)

    ######################### spec上下限选择
    Line_frame = tk.Frame(nova_cpp05_window)
    Line_frame.grid(row=3, column=2, columnspan=2, padx=10, pady=5, sticky='w')

    global upper_spec_entry, lower_spec_entry, upper_limit_entry, lower_limit_entry

    upper_limit_label = tk.Label(Line_frame, text="Upper Limit", font=("Arial", 14))
    upper_limit_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
    upper_limit_entry = tk.Entry(Line_frame)
    upper_limit_entry.grid(row=0, column=1, padx=5, pady=5)
    
    upper_spec_label = tk.Label(Line_frame, text="Upper Spec", font=("Arial", 14))
    upper_spec_label.grid(row=1, column=0, padx=5, pady=5, sticky='w')
    upper_spec_entry = tk.Entry(Line_frame)
    upper_spec_entry.grid(row=1, column=1, padx=5, pady=5)
    
    

    lower_spec_label = tk.Label(Line_frame, text="Lower Spec", font=("Arial", 14))
    lower_spec_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')
    lower_spec_entry = tk.Entry(Line_frame)
    lower_spec_entry.grid(row=2, column=1, padx=5, pady=5)

    lower_limit_label = tk.Label(Line_frame, text="Lower Limit", font=("Arial", 14))
    lower_limit_label.grid(row=3, column=0, padx=5, pady=5, sticky='w')
    lower_limit_entry = tk.Entry(Line_frame)
    lower_limit_entry.grid(row=3, column=1, padx=5, pady=5)
    
    ####################################
    

# Create a frame to hold the buttons
    button_frame = tk.Frame(nova_cpp05_window)
    button_frame.grid(row=4, column=1, padx=10, pady=(0,2), sticky="ew")

# Add buttons inside the frame
    add_button = tk.Button(button_frame, text="Add to Bucket", command=add_to_bucket)
    add_button.pack(side="left", expand=True, fill="x", padx=(0, 5),pady=(0,0))  # Left button

    remove_button = tk.Button(button_frame, text="Remove", command=remove_selected)
    remove_button.pack(side="left", expand=True, fill="x", padx=(5, 0),pady=(0,0))  # Right button
    
    
    plot_button = tk.Button(button_frame, text="Plot", command=lambda: plot_data_nova_cpp05())
    plot_button.pack(side=tk.LEFT, padx=5)
    # Back button
    back_button = tk.Button(nova_cpp05_window, text="Back", command=nova_cpp05_window.destroy)
    back_button.grid(row=6, column=1, columnspan=1, padx = (30,0),pady=10)

def update_datetime_combobox(event):
    selected_slot = slots_combobox.get()
    if df is not None and selected_slot:
        datetimes = df[df['Slot'] == int(selected_slot)]['DateTime'].unique()
        datetime_combobox['values'] = list(datetimes)
        datetime_combobox.set('')  # Clear previous selection

def add_to_bucket():
    selected_slot = slots_combobox.get()
    selected_datetime = datetime_combobox.get()
    if selected_slot and selected_datetime:
        bucket_listbox.insert(tk.END, f"Slot: {selected_slot}, DateTime: {selected_datetime}")
    else:
        messagebox.showerror("Error", "Both Slot and DateTime must be selected before adding to the bucket.")

def remove_selected():
    selected_indices = bucket_listbox.curselection()
    # Get the index of the selected item
    if selected_indices:
        for index in reversed(selected_indices):  # Iterate in reverse order to avoid index shifting issues
            bucket_listbox.delete(index)
         # Remove the selected item from the listbox
    else:
        messagebox.showerror("Error", "No item selected to remove.")  
        
def plot_data_nova_cpp05():
    bucket_listbox.select_set(0, tk.END)
    conditions = bucket_listbox.get(0, tk.END)
  
    if not conditions:
        messagebox.showerror("Error", "No conditions selected in the bucket.")
        return
    filtered_df = df.copy()
   
    y_label = label_combobox.get()
    plot_title = title_entry.get()
    try:
        upper_limit = float(upper_limit_entry.get())
        upper_spec = float(upper_spec_entry.get())
        lower_spec = float(lower_spec_entry.get())
        lower_limit = float(lower_limit_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Upper and Lower limits must be numbers")
        return

    global saved_y_label, saved_upper_limit, saved_lower_limit,saved_upper_spec, saved_lower_spec
    
    saved_y_label = y_label
    saved_upper_limit = upper_limit
    saved_lower_limit = lower_limit
    saved_upper_spec = upper_spec
    saved_lower_spec = lower_spec

    target = (upper_limit + lower_limit) / 2


    plt.figure(figsize=(14, 8))

    # Iterate through each condition in the bucket
    for condition in conditions:
        slot_condition, datetime_condition = condition.split(", ")
        slot = int(slot_condition.split(": ")[1])
        datetime = datetime_condition.split(": ")[1]

        # Filter data based on the current condition
        condition_df = filtered_df[(filtered_df['Slot'] == slot) & (filtered_df['DateTime'] == datetime)].copy()

        if condition_df.empty:
            continue

        # Calculate Radius (mm)
        condition_df['Radius (mm)'] = (condition_df['MpointX'].astype(float) / 1000) ** 2 + (condition_df['MpointY'].astype(float) / 1000) ** 2
        condition_df['Radius (mm)'] = condition_df['Radius (mm)'].apply(math.sqrt)

        # Plot the data for the current condition
        plt.scatter(condition_df['Radius (mm)'], condition_df["OX_HEIGHT"], label=f'Slot {slot}')

    # Add lines and text to the plot
    plt.axhline(y=upper_spec, color='r', linestyle='--', label='Upper Spec')
    plt.axhline(y=lower_spec, color='r', linestyle='--', label='Lower Spec')
    plt.axhline(y=target, color='r', linestyle='-', label='Target')
    plot_setting.add_text(110, lower_limit  + 100, "Z4")
    plot_setting.add_text(130, lower_limit  + 100, "Z3")
    plot_setting.add_text(140, lower_limit  + 100, "Z2")
    plot_setting.add_text(146, lower_limit  + 100, "Z1")
    
    plt.xlabel("Radius (mm)", fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.title(plot_title if plot_title else "Scatter Plot of OX_HEIGHT vs Radius", fontsize=24)
    plt.ylim([lower_limit, upper_limit])
    plt.legend(ncol = 10 , loc='center left', bbox_to_anchor=(0.1, -0.12))
    plt.xticks(range(0, 151, 15))
    plt.show()

    nova_cpp05_window.deiconify() 
           

##############################################################        
def analyze_Nova_CPO09_10():
    messagebox.showinfo("Info", "Nova(CPO09, CPO10) analysis selected")

def close_app():
    root.destroy()
# 创建按钮框架
frame = tk.Frame(root)
frame.pack(pady=20)

# 创建按钮
button_OCD = tk.Button(frame, text="OCD", command=analyze_OCD, width=20)
button_OCD.grid(row=0, column=0, padx=10, pady=10)

button_siview = tk.Button(frame, text="siview", command=analyze_siview, width=20)
button_siview.grid(row=0, column=1, padx=10, pady=10)

button_Nova_CPP05 = tk.Button(frame, text="Nova(CPP05)", command=analyze_Nova_CPP05, width=20)
button_Nova_CPP05.grid(row=1, column=0, padx=10, pady=10)

button_Nova_CPO09_10 = tk.Button(frame, text="Nova(CPO09, CPO10)", command=analyze_Nova_CPO09_10, width=20)
button_Nova_CPO09_10.grid(row=1, column=1, padx=10, pady=10)

# 创建关闭按钮
button_exit = tk.Button(frame, text="Exit", command=close_app, width=20)
button_exit.grid(row=2, column=0, columnspan=2, pady=20)

root.mainloop()





