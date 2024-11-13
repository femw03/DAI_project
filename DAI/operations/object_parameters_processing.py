import numpy as np
import cv2
from interfaces.interfaces import BoundingBox
from extract_depth_data import extract_depth

def calculate_object_distance(depth_image, bounding_box: BoundingBox):
    kernal_size = 5
    sigma = 1.5
    #Region of interest - bounding box coordinates
    x_min, x_max, y_min, y_max = bounding_box
    #Extract depth data
    depth_in_meters = extract_depth(depth_image)
    #Apply Gaussian smoothing to the entire depth map to reduce noise
    smoothed_depth_map = cv2.GaussianBlur(depth_in_meters, (kernal_size, kernal_size), sigma)
    #Extract the region of interest (bounding box) from the smoothed depth map
    roi_depth = smoothed_depth_map[y_min:y_max, x_min:x_max]
    #Apply median filtering to the ROI for salt-and-pepper noise reduction
    filtered_roi_depth = cv2.medianBlur(roi_depth.astype(np.float32), 3)
    #Find the darkest pixel (smallest depth value) in the filtered ROI
    min_val, min_loc = cv2.minMaxLoc(filtered_roi_depth)[:2]  # min_loc gives the (x, y) position within the ROI
    return min_val
