import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
print("The input should be: Attack or Defense or Sp_attack or Sp_defense or Speed")
choice = input("please input your choice: ")

df = pd.read_csv("Fire pokemon.csv")
line_wid = 4
x = df['pokemon'].tolist()
fig = plt.figure(figsize=(22,13))
color = ['#E0B30E','#8D918A','#D4A492','#D0D9C3']
fig,ax = plt.subplots(figsize=(22,13))

class decorate:
    def make_img(self,img,zoom, x, y):
        img = mpimg.imread(img)
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, (x,y),frameon=False)
        ax.add_artist(ab)    
        return
    def plot_bar(self,x,height,width,color):  
        plt.bar(x=x,height=height,width=width,color=color)
        plt.text(s=x,x=x,y=height/1.2,va='bottom',ha='center',fontsize=32*width)
        return
    def plot_rank(self,str,i,y):       
        plt.text(s=str,x=i,y=y+0.1,va='bottom',ha='center',font='Comic Sans MS',fontsize=40)
        return
    

def adv(element):
    y = [attack for attack in df[element]]
    width = [width/max(y) for width in y] 
    sort_width = list(sorted(set(width)))   
    d = decorate()
    for i in range(7):    
        if width[i] == sort_width[-1]: 
           d.make_img(img="pokemon figure/"+(df['pokemon'][i]) + '.png',zoom =0.12, x= x[i], y = 20)
           d.plot_bar(x[i],y[i],width[i],color[0])
           d.plot_rank("1st",i,y[i])
        if width[i] == sort_width[-2]:
           d.make_img(img = "pokemon figure/"+(df['pokemon'][i]) + '.png',zoom =0.12, x= x[i], y = 20)
           d.plot_bar(x[i],y[i],width[i],color[1])   
           d.plot_rank("2nd",i,y[i])            
        if width[i] == sort_width[-3]:
           d.make_img(img="pokemon figure/"+(df['pokemon'][i]) + '.png',zoom =0.12, x= x[i], y = 20)
           d.plot_bar(x[i],y[i],width[i],color[2])   
           d.plot_rank("3rd",i,y[i])                   
        if width[i] < sort_width[-3]:
           d.make_img(img= "pokemon figure/"+(df['pokemon'][i]) + '.png',zoom =0.12, x= x[i], y = 20)
           d.plot_bar(x[i],y[i],width[i],color[3])    
    plt.ylim(0,max(y)+50)
    plt.ylabel(element,fontsize = 50)
    plt.yticks(fontsize = 30)
    plt.xticks([])
    plt.tick_params(length = 8 ,width = 4)
    ax.spines['top'].set_linewidth(line_wid)
    ax.spines['bottom'].set_linewidth(line_wid)
    ax.spines['left'].set_linewidth(line_wid)
    ax.spines['right'].set_linewidth(line_wid)
    plt.savefig('result/'+str(choice)+'.png',dpi = 400)
    plt.pause(3)
    plt.close()
adv(choice)




