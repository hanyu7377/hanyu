import pandas as pd
import plotly.graph_objects as go
import numpy as np
df = pd.read_csv("cwurData.csv")
print(df)
####抓出年份在2014年的資料
df_2014 = df[df["year"] == 2014]
df_2015 = df[df["year"] == 2015]
year_2014 = go.Bar(x = df_2014["country"].unique(),
                y = df_2014['country'].value_counts().head(20),###列出前20名的國家
                name = "2014",
                marker = dict(color = 'rgba(255, 165, 0, 1)',###第四個數字是透明度
                line=dict(color='rgb(0,0,0)',width=2.0)),) ###控制box框的顏色 0,0,0代表黑色
year_2015 = go.Bar(x = df_2015["country"].unique(),
                y = df_2015['country'].value_counts().head(20),
                name = "2015",
                marker = dict(color = 'rgba(75, 0, 130, 1)',
                line=dict(color='rgb(0,0,0)',width=2.0)),)
data = [year_2014,year_2015]
layout = go.Layout(
    ####以CSS的語法加入<b>和</b>可以讓文字變粗體
    #####titlefont可以調整label字體大小   tickfont可以調整x軸上文字的大小
    title = dict(text = "<b>Top 1000 university distribution base on country</b>",font = dict(size = 50),x = 0.5),
    xaxis= dict(title = "<b>Country</b>",titlefont = dict(size = 30),tickfont=dict(size=20)),
    yaxis = dict(title = "<b>The number of university count</b>",titlefont = dict(size = 30),tickfont=dict(size=20)),
    legend=dict(x=0.8, y=1,  #####(1,1) 為右下角
    bgcolor='rgba(255, 255, 255, 0.5)',  ######圖例的背景顏色
    bordercolor='rgb(1, 1, 1)',  ######圖例的邊框顏色 1,1,1代表白色
    borderwidth=0,  ###### 圖例的邊框寬度
    font=dict(size=40)),###legend字體大小
    plot_bgcolor = "white", ######legend背景顏色
    hoverlabel=dict(font=dict(size=36))####互動式框的字體大小
)
fig = go.Figure(data = data,layout = layout)
fig.show()
                
