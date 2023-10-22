import pandas as pd
import numpy as np
import plotly.graph_objects as go

import plotly.express as px
df = pd.read_csv("../../Data/world_population.csv",encoding= "latin1")
fig = go.Figure()


grouper = df.groupby("Continent")["World Population Percentage"].sum().reset_index()
print(grouper)
fig = px.pie(grouper, 
             values="World Population Percentage", 
             names="Continent",
             title="Percentage of Population in Main Cities by Continent"
            )

fig.update_layout(title = dict(text = "<b>Percentage of Population in Main Cities by Continent</b>",
                               x = 0.5,
                               font = dict(size = 32)))
fig.update_layout(legend = dict(font = dict(size = 20)))
fig.update_traces(hovertemplate='%{label}: %{value:.2f}%') # 自定义悬停提示的内容)
fig.update_layout(hoverlabel=dict(font=dict(size=36)))
fig.update_traces(textfont=dict(size=30))
fig.write_html("pie_plot.html")
fig.show()