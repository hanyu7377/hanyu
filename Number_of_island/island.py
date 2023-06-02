import random
import numpy as np
import matplotlib.pyplot as plt
# Generate a random map
def map_generate(dimension):
    matrix_map = []
    for a in range(dimension):
        matrix_map.append([])
        for _ in range(dimension):
            element = random.choice([0, 1])
            matrix_map[a].append(element)         
    return matrix_map
map = map_generate(8)
# Get the land index in this part
is_land = []
for row in range(len(map)):
    for column in range(len(map[0])):
        if map[row][column] == 1:
            is_land.append([row, column])
# search function
def search(start_position):
    x = start_position[0]
    y = start_position[1]
    if [x, y+1] in is_land and [x, y +1] not in buffer:
        buffer.append([x, y+1])###search to right
    if [x, y-1] in is_land and [x, y-1] not in buffer:
        buffer.append([x, y-1])###search to left
    if [x-1, y] in is_land and [x-1, y] not in buffer:
        buffer.append([x-1, y])###search to top
    if [x+1, y] in is_land and [x+1, y] not in buffer:
        buffer.append([x+1, y])###search to down
buffer = []
count = 0
while is_land:
    print("The start position of this new island is:",is_land[0])
    start_point = is_land[0]
    buffer.append(start_point)
    print(buffer)
    area = 0
    while buffer:
        current_point = buffer.pop()#####peak the last element in buffer list and throw it
        if current_point in is_land:
            is_land.remove(current_point)
            area += 1
            search(current_point)
            print(area)
            print(buffer)
    if area > 0:
        count += 1
plt.imshow(map, cmap=plt.cm.binary)
plt.title("There are " + str(count)+ " islands in this map.",fontsize = 24)
plt.savefig('map.png')
plt.show()