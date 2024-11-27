"""
Contains helper functions to determine certain attributes from given depth information and bounding boxes
"""

from dataclasses import dataclass
from typing import List, Literal, Tuple

import cv2
import numpy as np

from ..interfaces import BoundingBox


@dataclass
class ObjectDistance:
    location: Tuple[float, float]
    depth: float


def calculate_object_distance(
    depth_image: np.ndarray, bounding_box: BoundingBox
) -> ObjectDistance:
    """
    Takes a depth image (width, height) and calculates the location (x,y)
    and value of the brightest pixel in the ROI defined by the bounding_box
    """
    kernal_size = 5
    sigma = 1.5
    # Region of interest - bounding box coordinates
    x_min, x_max, y_min, y_max = (
        bounding_box.x1,
        bounding_box.x2,
        bounding_box.y1,
        bounding_box.y2,
    )
    # Extract depth data
    depth_in_meters = depth_image  # TODO calculate depth_image to meters
    # Apply Gaussian smoothing to the entire depth map to reduce noise
    smoothed_depth_map = cv2.GaussianBlur(
        depth_in_meters, (kernal_size, kernal_size), sigma
    )
    # Extract the region of interest (bounding box) from the smoothed depth map
    roi_depth = smoothed_depth_map[y_min:y_max, x_min:x_max]
    # Apply median filtering to the ROI for salt-and-pepper noise reduction
    # filtered_roi_depth = cv2.medianBlur(roi_depth.astype(np.float32), 3)
    # Find the darkest pixel (smallest depth value) in the filtered ROI
    min_depth, max_depth, min_loc, max_loc = cv2.minMaxLoc(
        roi_depth
    )  # min_loc gives the (x, y) position within the ROI
    # Calculate the position of the closest point in the original image coordinates
    closest_x = x_min + min_loc[0]
    closest_y = y_min + min_loc[1]
    return ObjectDistance((closest_x, closest_y), min_depth)


def calculate_lateral_movement_by_optical_flow(
    prev_frame: np.ndarray, next_frame: np.ndarray, bounding_box: BoundingBox
) -> Literal["left", "right", "stationary"]:
    x_min, x_max, y_min, y_max = bounding_box

    # Convert frames to grayscale if they are in RGB
    if len(prev_frame.shape) == 3:
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    if len(next_frame.shape) == 3:
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Extract the region of interest (ROI) from both frames
    roi_prev = prev_frame[y_min:y_max, x_min:x_max]
    roi_next = next_frame[y_min:y_max, x_min:x_max]

    # Calculate optical flow within the bounding box
    flow = cv2.calcOpticalFlowFarneback(
        roi_prev, roi_next, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Calculate the average horizontal (x-direction) flow
    lateral_movement = np.mean(flow[:, :, 0])  # x-direction flow

    # Determine the lateral movement direction based on the sign of lateral_movement
    if lateral_movement < 0:
        lateral_direction = "left"
    elif lateral_movement > 0:
        lateral_direction = "right"
    else:
        lateral_direction = "stationary"

    return lateral_direction


# depth_frames: list of depth images
# bounding_boxes: list of bounding boxes [(x_min, y_min, x_max, y_max)] for each frame
# time_interval: time difference between frames (e.g., 1/fps)
def calculate_relative_motion(
    depth_frames: List[np.ndarray],
    bounding_boxes: List[BoundingBox],
    time_interval: float,
):
    motion_data = []

    for i in range(len(depth_frames) - 1):
        # Depth and closest point for current and next frames
        distance_i, closest_point_i = calculate_object_distance(
            depth_frames[i], bounding_boxes[i]
        )
        distance_next, closest_point_next = calculate_object_distance(
            depth_frames[i + 1], bounding_boxes[i + 1]
        )

        # Forward/backward speed calculation
        delta_distance = distance_next - distance_i
        speed = delta_distance / time_interval

        # Forward/backward direction
        if delta_distance < 0:
            forward_backward = "approaching"
        else:
            forward_backward = "receding"

        # Calculate lateral movement by x-coordinate change of the closest point
        delta_x = closest_point_next[0] - closest_point_i[0]
        if delta_x < 0:
            lateral = "left"
        elif delta_x > 0:
            lateral = "right"
        else:
            lateral = "stationary"

        # Or calculate lateral direction by optical flow
        # lateral = calculate_lateral_movement_by_optical_flow(depth_frames[i], depth_frames[i + 1], bounding_boxes[i])

        # Collect all motion data
        motion_data.append(
            {
                "frame": i,
                "distance": distance_i,
                "speed": speed,
                "direction": (forward_backward, lateral),
                "closest_point": closest_point_i,
            }
        )

    return motion_data


def calculate_anlge(x: float, FOV: float, width: float) -> float:
    """Calculates the angle in radians of the x coordinate from the center w.r.t. to FOV and width"""
    relative_x = (x - width / 2) / width  # the position of x w.r.t the center.
    FOV_in_radians = FOV * np.pi / 180
    angle = relative_x * FOV_in_radians
    return angle


def depth_image_to_meters(depth_image: np.ndarray) -> np.ndarray:
    # TODO move to NumpyLidar
    # Check the shape of the image
    if depth_image.shape[2] != 3:
        raise ValueError("The depth image must have 3 channels (RGB).")

    # Convert the RGB image to a depth map
    depth_normalized = (
        depth_image[:, :, 0]  # Red channel
        + depth_image[:, :, 1] * 256  # Green channel
        + depth_image[:, :, 2] * 256**2  # Blue channel
    ) / (256**3 - 1)  # Normalize to range [0, 1]

    # Convert normalized depth values to meters
    depth_in_meters = 1000 * depth_normalized  # Example conversion, adjust as necessary

    return depth_in_meters
