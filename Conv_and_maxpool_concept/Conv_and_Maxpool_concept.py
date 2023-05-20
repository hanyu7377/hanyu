import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
Im = Image.open('eight_10.png')
im_width = 151
im_height = 151
Oringinal_data = np.asarray(Im.resize((im_width,im_height)))
print(Oringinal_data.shape)
plt.imshow(Oringinal_data, cmap='gray_r', interpolation='none')
plt.subplots_adjust(bottom=0.1)
plt.title("Original Image",fontsize = 20)
plt.tight_layout()

plt.xlim(0,im_width )
plt.ylim(0,im_height)
plt.xlabel("PIXEL",size  =20)
plt.ylabel("PIXEL",size  =20)
plt.savefig('original.png', bbox_inches="tight")
plt.show()
my_filter = [[0.56,0.45],[0.87,0.15]]
print("the filter is: ", my_filter)
def convolution_layer_output(filter,input_image):    
    dimension = input_image.shape[0]-len(filter[0]) + 1
    print("the dimension of the output is: ", dimension)
    output = np.zeros((dimension,dimension))    
    for i in range(dimension * dimension):
        temp_sum = 0       
        start = [i//dimension,i%dimension]
        temp_sum += filter[0][0]* input_image[start[0],start[1] ]
        temp_sum += filter[0][1]* input_image[start[0] + 1, start[1] ]
        temp_sum += filter[1][0]* input_image[start[0],start[1] + 1]
        temp_sum += filter[1][1]* input_image[start[0] + 1 ,start[1] + 1]
        output[start[0]][start[1]] = temp_sum 
    return output   
def max_pooling(Conv_layer_output,width):   
    dimension = int(Conv_layer_output.shape[0] / width)###dimension = 100
    new_image = np.zeros((dimension,dimension))  
    for a in range(dimension): 
        for b in range(dimension):
            tmp_list = []
            start_position = [a * width, b * width ]
            tmp_list.append(Conv_layer_output[start_position[0]][start_position[1]])
            tmp_list.append(Conv_layer_output[start_position[0]+ 1][start_position[1]])
            tmp_list.append(Conv_layer_output[start_position[0]][start_position[1]+ 1])
            tmp_list.append(Conv_layer_output[start_position[0] + 1][start_position[1] + 1])
            new_image[a][b] = max(tmp_list)
    return new_image
convoluted_image = max_pooling(convolution_layer_output(my_filter,Oringinal_data),2)
plt.imshow(convoluted_image,cmap='gray_r', interpolation='none')
plt.title("Convolutioned Image",fontsize = 20)
plt.tight_layout()
plt.xlabel("PIXEL",size  =20)
plt.ylabel("PIXEL",size  =20)
plt.xlim(0.,convoluted_image.shape[0])
plt.ylim(0.,convoluted_image.shape[1])
plt.savefig('convolutioned.png', bbox_inches="tight")
plt.show()    
    


    
