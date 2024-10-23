import numpy as np
from interfaces.interfaces import BoundingBox

def calculate_object_distance(lidar_points_2d: np.array, bounding_box: BoundingBox):
    # Filter LiDAR points that fall within the bounding box
    object_points = lidar_points_2d[
        (lidar_points_2d[:, 0] >= bounding_box.x1) & (lidar_points_2d[:, 0] <= bounding_box.x2) &
        (lidar_points_2d[:, 1] >= bounding_box.y1) & (lidar_points_2d[:, 1] <= bounding_box.y2) 
    ]

    # Compute the centroid of the object
    object_centroid = object_points.mean(axis=0)  # or np.median(object_points, axis=0)

    # Calculate Euclidean distance from sensor origin (0, 0, 0)
    sensor_origin = np.array([0.0, 0.0])
    distance = np.linalg.norm(object_centroid - sensor_origin)

    print(f"Distance to the object: {distance:.2f} meters")
    return distance

