import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
df = pd.read_csv("../../Data/world_population.csv",encoding = "latin1")
fig = go.Figure()

def plot(city,color):
    print("City:", city)
    print("Available cities in DataFrame:", df["Capital"].unique())
    print("City data:", city)
    year = list(map(str, range(1970, 2021, 10)))
    #####obtain population data
    population = df.loc[df["Capital"] == str(city)].values.tolist()[0][6:13][::-1]
    fig.add_trace(go.Scatter(x=year, y=population,
                         mode="markers",
                         marker=dict(size=population,
                                     sizemode="area",
                                     sizeref=2. * max(population) / (30. ** 2),                                
                                     colorscale = color,
                                     showscale=False),
                         name=str(city) 
                         ))
    return
plot("Tokyo","Reds")
plot("Ottawa","Greens")
plot("Seoul","Blues")
plot("Washington, D.C.","plasma")
plot("Moscow","electric")
plot("Taipei","ice")
plot("London","hot")
plot("Rome","purples")
plot("Berlin","speed")
fig.update_layout(
    title=dict(text="<b>Population Trend in Big city</b>", font=dict(size=50), x=0.5),
    xaxis=dict(title="<b>Year</b>", titlefont=dict(size=30), tickfont=dict(size=20)),
    yaxis=dict(title="<b>Population</b>", titlefont=dict(size=30), tickfont=dict(size=20)),
    plot_bgcolor="White",
    hoverlabel=dict(font=dict(size=36))
)
fig.write_html("scatter_plot.html")
fig.show()
