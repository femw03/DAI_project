import numpy as np
from distance_operations import  filter_image_noise, add_noise, convert_maxLoc_bb, calculate_object_distance_bounding_box, calculate_object_distance_maxLoc, process_image
from ..interfaces import BoundingBox
import cv2
import random

# Example LiDAR data (x, y points in 2D space)
lidar_points_2d = np.array([
    [4.0, 32.0],  # Point at (x=1.0, y=2.0)
    [20.5, 4.5],  # Point at (x=1.5, y=3.5)
    [31.0, 9.0],  # Point at (x=2.0, y=2.2)
    [30.0, 5.0],  # Point at (x=2.5, y=1.5)
    [19.0, 4.0]   # Point at (x=3.0, y=3.0)
])

# Define bounding box range in 2D (xmin, xmax, ymin, ymax, zmin, zmax)
#bounding_box = BoundingBox(x1=0.5, x2=10.5, y1=1.5, y2=2.5)
#calculate_object_distance(lidar_points_2d, bounding_box)
# Load the image
image_path = 'D:/Soja/1_Masters/AI Lab/AI project/Images/cropped_image.png'
#Read colour image
colour_img =cv2.imread(image_path, cv2.IMREAD_COLOR)
print(type(colour_img))
maxLoc = process_image(colour_img)
#Calculate distance
bounding_box = convert_maxLoc_bb(maxLoc)
print(F"bounding box {bounding_box.x1},{bounding_box.x2},{bounding_box.y1},{bounding_box.y2}")
##To decide which of the below methods to be used
calculate_object_distance_bounding_box(lidar_points_2d, bounding_box)
calculate_object_distance_maxLoc(lidar_points_2d, maxLoc)

##Sample code for noise filtering
def filter_image_noise(image):
    S_max = 3# Maximum window size
    padded_image = np.pad(image, S_max // 2, mode='constant', constant_values=0)
    output_image = np.copy(image)
    rows, cols  = image.shape
    for i in range(rows):
        for j in range(cols):
            S = 3
            while S <= S_max:
                # Extracting the subimage
                sub_img = padded_image[i:i + S, j:j + S]
                Z_min = np.min(sub_img)
                Z_max = np.max(sub_img)
                Z_m = np.median(sub_img)
                Z_xy = image[i, j]

                if Z_min < Z_m < Z_max:
                    if Z_min < Z_xy < Z_max:
                        output_image[i, j] = Z_xy
                    else:
                        output_image[i, j] = Z_m
                    break
                else:
                    S += 2
            else:
                output_image[i, j] = Z_m

    return output_image

#For testing purpose
def add_noise(img): 
  
    # Getting the dimensions of the image 
    rows, cols  = img.shape
     
    # Randomly pick some pixels in the image for coloring them white 
    # Pick a random number between 300 and 10000 
    number_of_pixels = random.randint(300, 10000) 
    for i in range(number_of_pixels): 
        
        # Pick a random y coordinate 
        y_coord=random.randint(0, rows - 1) 
          
        # Pick a random x coordinate 
        x_coord=random.randint(0, cols - 1) 
          
        # Color that pixel to white 
        img[y_coord][x_coord] = 255
          
    # Randomly pick some pixels in the image for coloring them black 
    # Pick a random number between 300 and 10000 
    number_of_pixels = random.randint(300 , 10000) 
    for i in range(number_of_pixels): 
        
        # Pick a random y coordinate 
        y_coord=random.randint(0, rows - 1) 
          
        # Pick a random x coordinate 
        x_coord=random.randint(0, cols - 1) 
          
        # Color that pixel to black 
        img[y_coord][x_coord] = 0
          
    return img 