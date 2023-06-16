import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import base64
##########Add description of this dashboard
st.set_page_config(page_title="Pokemon analysis dashboard", page_icon="pokemon.png", layout="wide" )
st.markdown('<h2 style="text-align: center;font-size: 70px">Welcome to Pokemon world !!!</h2>', unsafe_allow_html=True)
st.markdown('<h2 style="text-align: center;font-size: 30px">This app performs some basic analysis with pokemon state raw data from Kaggle</h2>', unsafe_allow_html=True)
st.markdown('<h2 style="text-align: center; font-size: 30px;">Data source: <a href="https://www.kaggle.com/datasets/abcsds/pokemon" target="_blank">https://www.kaggle.com/datasets/abcsds/pokemon</a></h2>', unsafe_allow_html=True)
######show gif
file_ = open("icegif-1065.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
    unsafe_allow_html=True,
)
###################

##########Load raw data and show it on the dashboard
df = pd.read_csv("Pokemon.csv", encoding='latin1', dtype={'last_column': bool})
df['Legendary'] = df['Legendary'].map({True: 'TRUE', False: 'FALSE'})

st.write(df)




st.write("Explanation in each attribute")

attribute = {'ID' : 'ID for each pokemon',
'Name': 'Name of each pokemon',
'Type 1': 'Each pokemon has a type, this determines weakness/resistance to attacks',
'Type 2': 'Some pokemon are dual type and have 2',
'Total': 'sum of all stats that come after this, a general guide to how strong a pokemon is',
'HP': 'hit points, or health, defines how much damage a pokemon can withstand before fainting',
'Attack': 'the base modifier for normal attacks (eg. Scratch, Punch)',
'Defense': 'the base damage resistance against normal attacks',
'SP Atk': 'special attack, the base modifier for special attacks (e.g. fire blast, bubble beam)',
'SP Def': 'the base damage resistance against special attacks',
'Speed': 'determines which pokemon attacks first each round'}
for attri , mean in attribute.items():
    st.write(f"<li><b>{attri}:</b> {mean}</li>",unsafe_allow_html=True)
#################################for first analysis

class decorate_with_figure:
    def get_figure(self,first,second,third):
        first = Image.open(first)
        second = Image.open(second)
        third = Image.open(third)
        col1, col2, col3,col4,col5 = st.columns(5)
        with col1:
            st.image([first])
        with col2:
            st.write(' ')
        with col3:
            st.image([second])
        with col4:
            st.write(' ')
        with col5:
            st.image([third])
decorate = decorate_with_figure()       
class filter:
    def filter_title(self,title):
        st.sidebar.header(title)
        return
    def multiple_selection(self,header_title,filter,key):
        result = st.sidebar.multiselect(header_title, df[str(filter)].unique(),key=key)
        return result
    def single_selection(self,header_title,filter,key):
        result = st.sidebar.selectbox(header_title, df[str(filter)].unique(),key=key)
        return result
filter = filter()
color_map = {
    'Bug': (166/255, 206/255, 57/255, 1),
    'Dark': (79/255, 74/255, 72/255, 1),
    'Dragon': (111/255, 53/255, 252/255, 1),
    'Electric': (252/255, 195/255, 77/255, 1),
    'Fairy': (232/255, 120/255, 144/255, 1),
    'Fighting': (193/255, 47/255, 47/255, 1),
    'Fire': (252/255, 95/255, 53/255, 1),
    'Flying': (159/255, 144/255, 246/255, 1),
    'Ghost': (123/255, 78/255, 155/255, 1),
    'Grass': (102/255, 205/255, 170/255, 1),
    'Ground': (226/255, 191/255, 101/255, 1),
    'Ice': (136/255, 206/255, 226/255, 1),
    'Normal': (166/255, 166/255, 139/255, 1),
    'Poison': (168/255, 78/255, 160/255, 1),
    'Psychic': (251/255, 93/255, 177/255, 1),
    'Rock': (183/255, 159/255, 52/255, 1),
    'Steel': (183/255, 183/255, 206/255, 1),
    'Water': (69/255, 146/255, 196/255, 1)
}
class plot_settings:
    def egde_of_plot(self,line_wid):
        ax.spines['top'].set_linewidth(line_wid)
        ax.spines['bottom'].set_linewidth(line_wid)
        ax.spines['left'].set_linewidth(line_wid)
        ax.spines['right'].set_linewidth(line_wid)
        plt.tick_params(length = 8 ,width = line_wid)
        return
    def label_of_plot(self,x,y):
        plt.xlabel(str(x),fontsize = 16)
        plt.ylabel(str(y),fontsize = 16)
        return
    def title_of_plot(self,title):
        plt.title(title,fontsize = 20,y = 1.05)
        return
    def color(self,color_map,selection):
        color_list = color_map.get(selection)
        return color_list
    def lim_of_plot(self,min_x,max_x,min_y,max_y):
        plt.xlim(min_x,max_x)
        plt.ylim(min_y,max_y)
        return
    def rotation(self,rotation):
        plt.xticks(rotation = rotation)
    def plot(self,fig):
        st.pyplot(fig)
        return
plot_settings = plot_settings()
decorate.get_figure("pokemon_jpg/1.jpg","pokemon_jpg/4.jpg","pokemon_jpg/7.jpg")
# 绘制第一只宝可梦的雷达图
filter.filter_title("Compare two pokemon species strenght with radar chart:  ")
first_pokemon_type_choice = filter.single_selection("Select type for 1st pokemon (according to Type 1)","Type 1","first pokemon")
fig, ax = plt.subplots(figsize=(10,10), subplot_kw={'projection': 'polar'})
if first_pokemon_type_choice:
    filtered_data = df[df["Type 1"].isin([first_pokemon_type_choice])]
    pokemon_name_first = st.sidebar.selectbox("Select first Pokemon", filtered_data['Name'])
    filtered_data_first = filtered_data[filtered_data['Name'] == pokemon_name_first]
if first_pokemon_type_choice and filtered_data_first is not None:
    values_first = filtered_data_first[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].values.tolist()[0]
    values_first.append(values_first[0])  # 添加第一个点的坐标，形成闭合的六边形
theta = np.linspace(0, 2*np.pi, len(values_first[:]))
ax.fill(theta, values_first[:7], color=color_map.get(first_pokemon_type_choice), alpha=0.3)
ax.plot(theta, values_first[:7], color=color_map.get(first_pokemon_type_choice), linewidth=3)
# 选择第二只宝可梦的类型和名称


selected_type_for_second_pokemon = st.sidebar.selectbox("Select type for 2nd pokemon (according to Type 1)", df['Type 1'].unique(),key = "second pokemon")
if selected_type_for_second_pokemon:
    filtered_data = df[df["Type 1"].isin([selected_type_for_second_pokemon])]
    pokemon_name_second = st.sidebar.selectbox("Select second Pokemon", filtered_data['Name'])
    filtered_data_second = filtered_data[filtered_data['Name'] == pokemon_name_second]
if selected_type_for_second_pokemon and filtered_data_second is not None:
    values_second = filtered_data_second[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].values.tolist()[0]
    values_second.append(values_second[0])  # 添加第一个点的坐标，形成闭合的六边形
    
theta = np.linspace(0, 2*np.pi, len(values_first[:]))
ax.fill(theta, values_second[:7], color=color_map.get(selected_type_for_second_pokemon), alpha=0.3)
ax.plot(theta, values_second[:7], color=color_map.get(selected_type_for_second_pokemon), linewidth=3)
# 設定子圖



# 繪製第二隻寶可夢的雷達圖

labels =['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def','Speed',"HP"]
# 設定雷達圖的範圍和刻度
ax.set_ylim(0, 200)
ax.set_yticks(np.arange(0, 201, 50))
ax.set_xticks(theta[:])
ax.set_xticklabels(labels[:])
ax.set_rlabel_position(1.2)
ax.tick_params(axis='y', labelsize=10)

ax.legend(labels=[pokemon_name_first, pokemon_name_second], loc='best')

plot_settings.title_of_plot('Two pokemon base stats comparison')
plot_settings.plot(fig)

##################################################################
###################################################################
###########start to plot bar chart to count pokemon by type 1#######
decorate.get_figure("pokemon_jpg/152.jpg","pokemon_jpg/155.jpg","pokemon_jpg/158.jpg")
filter.filter_title("Count pokemon based on Type 1 with bar chart")
def count_pokemon_by_type(df, types):
    return df[df["Type 1"].isin(types)]["Type 1"].value_counts()
choice = filter.multiple_selection("Select type (according to Type 1)","Type 1","count with type 1")
count = count_pokemon_by_type(df,choice)
color_list = [color_map[type] for type in count.index]
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(count.index, count.values, color=color_list)
plot_settings.title_of_plot('Number of Pokemon by Type 1')
plot_settings.egde_of_plot(3)
plot_settings.label_of_plot("Type 1","Count")
plot_settings.rotation(45)
plot_settings.plot(fig)




######
###########start to plot bar chart to count pokemon by generation
decorate.get_figure("pokemon_jpg/252.jpg","pokemon_jpg/255.jpg","pokemon_jpg/258.jpg")
#st.markdown('<h2 style="text-align: center;color: red;font-size: 50px">Count pokemon based on generation</h2>', unsafe_allow_html=True)
filter.filter_title("Count pokemon based on generation with bar chart:  ")
def count_pokemon_by_generation(df, generation):
    return df[df["Generation"].isin(generation)]["Generation"].value_counts()
choice = filter.multiple_selection("Select generation","Generation","count with generation")
count = count_pokemon_by_generation(df,choice)
color_list = ["Blue","Red","Green","Purple","Brown","Yellow"]
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(count.index, count.values, color=color_list)
plot_settings.egde_of_plot(3)
plot_settings.title_of_plot('Number of Pokemon by generation')
plot_settings.label_of_plot('Generation',"Count")
plot_settings.rotation(45)
plot_settings.plot(fig)



#########start to plot scatter plot
decorate.get_figure("pokemon_jpg/387.jpg","pokemon_jpg/390.jpg","pokemon_jpg/393.jpg")
fig,ax = plt.subplots(figsize=(8, 6))
filter.filter_title("Select two base stats and visulizae the trend by scatter plot: ")
type = filter.single_selection("Select type (according to Type 1)", 'Type 1',"type")
x = st.sidebar.selectbox("Select first base stats", df.columns[6:11],key="x")
y = st.sidebar.selectbox("Select second base stats", df.columns[6:11],key="y")
if x :
    filter_by_type  =  df[df["Type 1"].isin([type])]
    x_data = filter_by_type[str(x)]
if y :
    filter_by_type  =  df[df["Type 1"].isin([type])]
    y_data = filter_by_type[str(y)]
plt.scatter(x_data,y_data,marker="o",color = color_map.get(type))
plot_settings.egde_of_plot(3)
plot_settings.label_of_plot(str(x),str(y))
plot_settings.title_of_plot(str(x)+ " V.S " + str(y) + " in type " + str(type))
plot_settings.lim_of_plot(0,200,0,200)
plot_settings.plot(fig)