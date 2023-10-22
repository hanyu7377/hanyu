import plotly.graph_objects as go
import pandas as pd
import numpy as np
df = pd.read_csv("../../Data/timesData.csv",encoding = "latin1")
df.columns = df.columns.str.title()
print(df.columns)
fig = go.Figure()
def plot_box(y,name,color):
    fig.add_traces(go.Box(y= y,
                 name = name,
                 marker = dict(
                        color = color),
                 text=df['University_Name'],
                 ))
    return 
plot_box(df["Teaching"],"Teaching","blue")
plot_box(df["International"],"International","red")
plot_box(df["Research"],"Research","black")
plot_box(df["Citations"],"Citations","yellow")
plot_box(df["Total_Score"],"Total Score","green")
plot_box(df["Num_Students"],"Num Students","purple")
plot_box(df["Student_Staff_Ratio"],"Student Staff Ratio","cadetblue")
fig.update_layout(
    title=dict(text="<b>Analysis in university reputation</b>", font=dict(size=50), x=0.5),
    xaxis=dict(title="<b>Catogories</b>", titlefont=dict(size=30), tickfont=dict(size=24)),
    yaxis=dict(title="<b>Values</b>", titlefont=dict(size=30), tickfont=dict(size=24)),
    plot_bgcolor="White",
    hoverlabel=dict(font=dict(size=24))
)
fig.write_html("Box_plot.html")
fig.show()