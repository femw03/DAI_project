import numpy as np
import cv2
from interfaces.interfaces import BoundingBox

def calculate_object_distance_bounding_box(lidar_points_2d: np.array, bounding_box: BoundingBox):
    # Filter LiDAR points that fall within the bounding box
    object_points = lidar_points_2d[
        (lidar_points_2d[:, 0] >= bounding_box.x1) & (lidar_points_2d[:, 0] <= bounding_box.x2) &
        (lidar_points_2d[:, 1] >= bounding_box.y1) & (lidar_points_2d[:, 1] <= bounding_box.y2) 
    ]
    print(f"object points : {object_points}")
    # Compute the centroid of the object
    object_centroid = object_points.mean(axis=0)  # or np.median(object_points, axis=0)

    # Calculate Euclidean distance from sensor origin (0, 0, 0)
    sensor_origin = np.array([0.0, 0.0])
    distance = np.linalg.norm(object_centroid - sensor_origin)

    print(f"Distance to the object: {distance:.2f} meters")
    return distance

def convert_maxLoc_bb(maxLoc: tuple):
    horizontal_padding = 5
    vertical_padding = 5
    # Define the bounding box based on maxLoc
    xmax = maxLoc[0] + horizontal_padding
    xmin = max(maxLoc[0] - horizontal_padding,0)
    ymax = maxLoc[1] + vertical_padding
    ymin = max(maxLoc[1] - vertical_padding,0)
    bounding_box = BoundingBox(x1=xmin, x2=xmax, y1=ymin, y2=ymax)
    print(bounding_box)
    return bounding_box

def calculate_object_distance_maxLoc(lidar_points_2d: np.array, maxLoc: tuple):
    # Example maxLoc from cv2.minMaxLoc (coordinates of maximum intensity point)
    #maxLoc = (50, 100)  # (x, y) coordinates of the max intensity point

    # Convert maxLoc to a numpy array for vectorized computation
    maxLoc_array = np.array(maxLoc)

    # Calculate the Euclidean distance from each LiDAR point to maxLoc
    distances = np.linalg.norm(lidar_points_2d - maxLoc_array, axis=1)
    print(f"distances: {distances}")
    # Find the index of the minimum distance
    closest_index = np.argmin(distances)
    print(f"closest index: {closest_index}")
    # Get the LiDAR point corresponding to maxLoc
    closest_lidar_point = lidar_points_2d[closest_index]
    print(f"closest lidar point: {closest_lidar_point}")
    # Calculate Euclidean distance from sensor origin (0, 0)
    sensor_origin = np.array([0.0, 0.0])
    distance = np.linalg.norm(closest_lidar_point - sensor_origin)

    print(f"Distance to the object: {distance:.2f} meters")
    return distance

def show_image(img1, text1, img2, text2, img3, text3):
    cv2.imshow(text1, img1)
    cv2.imshow(text2,img2)
    cv2.imshow(text3,img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image(image):
    # Convert to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = gray_img.copy()
    #show_image(img, 'Original Image')
    #Add noise to grey image
    #noise_grey_image = add_noise(grey_img)
    # Apply adaptive median filtering
    #filtered_grey_img = filter_image_noise(noise_grey_image)
    # Apply median filtering
    filtered_gray_img = cv2.medianBlur(gray_img, 5)  # Use a 5x5 kernel
    # Apply smoothening (LPF)
    #filtered_grey_img = cv2.blur(noise_grey_image,(3,3))
    #filtered_grey_img = cv2.boxFilter(noise_grey_image, -1, (3,3))
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(filtered_gray_img)
    print(f"minVal: {minVal}")
    print(f"maxVal: {maxVal}")
    print(f"minLoc: {minLoc}")
    print(f"maxLoc: {maxLoc}")
    # display the results
    #cv2.circle(image, maxLoc, 7, (255, 0, 0), 2)
    #show_image(grey_img, 'grey', filtered_grey_img, 'filtered grey',image, 'final')
    return maxLoc
