import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image


df = pd.read_csv("Pokemon.csv", encoding='latin1', dtype={'last_column': bool})

st.set_page_config(page_title="Pokemon analysis dashboard", page_icon="pokemon.png", layout="wide" )

df['Legendary'] = df['Legendary'].map({True: 'TRUE', False: 'FALSE'})
st.write(df)

st.sidebar.header("Please filter here:")

# 选择第一只宝可梦的类型和名称
selected_type_for_first_pokemon = st.sidebar.selectbox("Select type for 1st pokemon (according to Type 1)", df['Type 1'].unique(),key="first_pokemon")

if selected_type_for_first_pokemon:
    filtered_data = df[df["Type 1"].isin([selected_type_for_first_pokemon])]
    pokemon_name_first = st.sidebar.selectbox("Select first Pokemon", filtered_data['Name'])
    filtered_data_first = filtered_data[filtered_data['Name'] == pokemon_name_first]

# 选择第二只宝可梦的类型和名称
selected_type_for_second_pokemon = st.sidebar.selectbox("Select type for 2nd pokemon (according to Type 1)", df['Type 1'].unique(),key = "second_pokemon")
if selected_type_for_second_pokemon:
    filtered_data = df[df["Type 1"].isin([selected_type_for_second_pokemon])]
    pokemon_name_second = st.sidebar.selectbox("Select second Pokemon", filtered_data['Name'])
    filtered_data_second = filtered_data[filtered_data['Name'] == pokemon_name_second]

# 创建雷达图
fig = go.Figure()
color_map = {
    'Bug': 'rgb(166,206,57)',
    'Dark': 'rgb(79,74,72)',
    'Dragon': 'rgb(111,53,252)',
    'Electric': 'rgb(252,195,77)',
    'Fairy': 'rgb(232,120,144)',
    'Fighting': 'rgb(193,47,47)',
    'Fire': 'rgb(252,95,53)',
    'Flying': 'rgb(159,144,246)',
    'Ghost': 'rgb(123,78,155)',
    'Grass': 'rgb(102,205,170)',
    'Ground': 'rgb(226,191,101)',
    'Ice': 'rgb(136,206,226)',
    'Normal': 'rgb(166,166,139)',
    'Poison': 'rgb(168,78,160)',
    'Psychic': 'rgb(251,93,177)',
    'Rock': 'rgb(183,159,52)',
    'Steel': 'rgb(183,183,206)',
    'Water': 'rgb(69,146,196)'
}

# 绘制第一只宝可梦的雷达图
if selected_type_for_first_pokemon and filtered_data_first is not None:
    values_first = filtered_data_first[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].values.tolist()[0]
    values_first.append(values_first[0])  # 添加第一个点的坐标，形成闭合的六边形
    theta = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'HP']
    type1 = filtered_data_first['Type 1'].iloc[0]
    fig.add_trace(go.Scatterpolar(
        r=values_first,
        theta=theta,
        fill='toself',
        line=dict(width=3, color=color_map.get(type1)),
        name=pokemon_name_first,
        text=theta,
        textfont=dict(size=400)
    ))

# 绘制第二只宝可梦的雷达图
if selected_type_for_second_pokemon and filtered_data_second is not None:
    values_second = filtered_data_second[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].values.tolist()[0]
    values_second.append(values_second[0])  # 添加第一个点的坐标，形成闭合的六边形
    type1 = filtered_data_second['Type 1'].iloc[0]
    fig.add_trace(go.Scatterpolar(
        r=values_second,
        theta=theta,
        fill='toself',
        line=dict(width=3, color=color_map.get(type1)),
        name=pokemon_name_second,
        text=theta,
        textfont=dict(size=400)
    ))

# 设置雷达图布局
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 200],
            tickfont=dict(size=20),
            tickformat="d"
        ),
        angularaxis=dict(direction="clockwise")
    ),
    showlegend=True,
    legend=dict(x=1, y=1, font=dict(size=30)),
    colorway=list(color_map.values()),
    plot_bgcolor='black',

)

# 在Streamlit的主界面显示雷达图
# def load_image(img):
#     im = Image.open(img)
#     image = np.array(im)
#     return image

# # Uploading the File to the Page
# uploadFile1 = st.file_uploader( "The 1st pokemon looking",type=['jpg', 'png'])
# uploadFile2 = st.file_uploader( "The 2nd pokemon looking",type=['jpg', 'png'])

# # Checking the Format of the page
# if uploadFile1 is not None:
#     # Perform your Manupilations (In my Case applying Filters)
#     img = load_image(uploadFile1)
#     st.image(img)
#     st.write("Image Uploaded Successfully")
# else:
#     st.write("Make sure you image is in JPG/PNG Format.")
# if uploadFile2 is not None:
#     # Perform your Manupilations (In my Case applying Filters)
#     img = load_image(uploadFile2)
#     st.image(img)
#     st.write("Image Uploaded Successfully")



# 在Streamlit的主界面显示雷达图
# 将自定义CSS文件链接到应用程序
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")
st.plotly_chart(fig)
