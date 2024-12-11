#!/usr/bin/env python

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import glob
import os
import sys
import shutil
import os.path
import time
import cv2

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- directories ---------------------------------------------------------------
# ==============================================================================

# Directories to create/empty
dirs = ['my_data/', 'draw_bounding_box/', 'custom_labels/']

for directory in dirs:
    if os.path.exists(directory):
        # Remove all contents in the directory
        shutil.rmtree(directory)
    # Create the (empty) directory
    os.makedirs(directory)


# ==============================================================================
# -- global variables ----------------------------------------------------------
# ==============================================================================

dataEA = len(next(os.walk('SegmentationImage/'))[2])

VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 90

# Segmentation colors (BGR): https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
CAR_BBOX_COLOR = (0, 0, 255)
PEDESTRIAN_BBOX_COLOR = (255, 0, 0)
TRAFFICLIGHT_BBOX_COLOR = (102, 255, 102)
TRAFFICSIGN_BBOX_COLOR = (255, 0, 255)
BUS_BBOX_COLOR = (0, 255, 255)
MOTORCYCLE_BBOX_COLOR = (0, 153, 0)
BICYCLE_BBOX_COLOR = (255, 255, 0)
TRUCK_BBOX_COLOR = (255, 255, 255)
TRAIN_BBOX_COLOR = (0, 128, 255)
RIDER_BBOX_COLOR = (255, 0, 127)

Car_COLOR = np.array([142, 0, 0])
Walker_COLOR = np.array([60, 20, 220])
TrafficLight_COLOR = np.array([30, 170, 250])
TrafficSign_COLOR = np.array([0, 220, 220])
Bus_COLOR = np.array([100, 60, 0])
Motorcycle_COLOR = np.array([230, 0, 0])
Bicycle_COLOR = np.array([32, 11, 119])
Truck_COLOR = np.array([70, 0, 0])
Train_COLOR = np.array([100, 80, 0])
Rider_COLOR = np.array([0, 0, 255])

car_class = 0
walker_class = 1
traffic_light_class = 2
traffic_sign_class = 3
bus_class = 4
motorcycle_class = 5
bicycle_class = 6
truck_class = 7
train_class = 8
rider_class = 9   

rgb_info = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype="i")
seg_info = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype="i")
area_info = np.zeros(shape=[VIEW_HEIGHT, VIEW_WIDTH, 3], dtype=np.uint8)
index_count = 0

# ==============================================================================
# -- classification functions --------------------------------------------------
# ==============================================================================

# Brings Images and Bounding Box Information
def reading_data(index):
    global rgb_info, seg_info

    rgb_img = cv2.imread('custom_data/image'+ str(index)+ '.png', cv2.IMREAD_COLOR)
    seg_img = cv2.imread('SegmentationImage/seg'+ str(index)+ '.png', cv2.IMREAD_COLOR)

    if str(rgb_img) != "None" and str(seg_img) != "None":
        origin_rgb_info = rgb_img
        rgb_info = rgb_img
        seg_info = seg_img
        return True

    else:
        return False


def write_bounding_box(f, class_id, darknet_x, darknet_y, darknet_width, darknet_height):
    f.write(f"{class_id} {darknet_x:.6f} {darknet_y:.6f} {darknet_width:.6f} {darknet_height:.6f}\n")

def process_object(mask_color, bbox_color, class_id, file): 
    global seg_info, area_info
       
    # Create a binary mask for the object
    mask = np.zeros(seg_info.shape[:2], dtype=np.uint8)
    mask[(seg_info[:, :, 0] == mask_color[0]) & 
         (seg_info[:, :, 1] == mask_color[1]) & 
         (seg_info[:, :, 2] == mask_color[2])] = 255

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 50:  # Filter small contours
            continue

        # Get bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Draw the bounding box on the original image
        cv2.rectangle(rgb_info, (x, y), (x + w, y + h), bbox_color, 2)

        # Calculate the darknet format values
        darknet_x = float(x + w // 2) / float(VIEW_WIDTH)
        darknet_y = float(y + h // 2) / float(VIEW_HEIGHT)
        darknet_width = float(w) / float(VIEW_WIDTH)
        darknet_height = float(h) / float(VIEW_HEIGHT)

        # Write the bounding box data to the text file
        write_bounding_box(file, class_id, darknet_x, darknet_y, darknet_width, darknet_height)


def run():
    global rgb_info
    global seg_info

    for i in range(dataEA):
        if reading_data(i+1) != False:
            f = open(f"custom_labels/image{i+1}.txt", 'a')
            
            process_object(Car_COLOR, CAR_BBOX_COLOR, car_class, f)
            process_object(Walker_COLOR, PEDESTRIAN_BBOX_COLOR, walker_class, f)
            process_object(TrafficLight_COLOR, TRAFFICLIGHT_BBOX_COLOR, traffic_light_class, f)
            process_object(TrafficSign_COLOR, TRAFFICSIGN_BBOX_COLOR, traffic_sign_class, f)
            process_object(Bus_COLOR, BUS_BBOX_COLOR, bus_class, f)
            process_object(Motorcycle_COLOR, MOTORCYCLE_BBOX_COLOR, motorcycle_class, f)
            process_object(Bicycle_COLOR, BICYCLE_BBOX_COLOR, bicycle_class, f)
            process_object(Truck_COLOR, TRUCK_BBOX_COLOR, truck_class, f)

            f.close()
            cv2.imwrite(f'draw_bounding_box/image{i+1}.png', rgb_info)
            
            print(i)

if __name__ == "__main__":
    start = time.time()

    run()

    end = time.time()
    print(float(end - start))
