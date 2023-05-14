import numpy as np
import matplotlib.pyplot as plt
import random 
my_list = list(random.sample(range(1,300),60))
print(my_list)
x = [x*8 for x in range(len(my_list))]
y = [0]*len(my_list)
z = [0]*len(my_list)
dx = [4]*len(my_list)
dy = [2]*len(my_list)
for i in range(len(my_list)-1):
    min_value_index = my_list.index(min(my_list[i:]))
    my_list[i],my_list[min_value_index] = my_list[min_value_index],my_list[i]

    dz = my_list
   

    ax = plt.axes(projection = '3d')


    ax.set_ylim(-1,10)
    
    ax.set_axis_off()
    ax.bar3d(x,y,z,dx,dy,dz,color = 'blue')
    
    plt.savefig("The" + str(i+1)+"th step"+".png",dpi = 1600)

    plt.pause(0.2)
    plt.close()
    if i == len(my_list)-1:
        
        plt.savefig("The" + str(i+1)+"th step"+".png",dpi = 1600)
        plt.pause(10)
        plt.close()
        
print(my_list)






