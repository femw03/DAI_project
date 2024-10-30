import cv2
import numpy as np
import argparse

def load_depth_image(image_path):
    # Load the RGB depth image
    depth_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if depth_image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    return depth_image

def extract_depth(depth_image):
    # Check the shape of the image
    if depth_image.shape[2] != 3:
        raise ValueError("The depth image must have 3 channels (RGB).")
    
    # Convert the RGB image to a depth map
    depth_normalized = (
        depth_image[:, :, 0] +    # Red channel
        depth_image[:, :, 1] * 256 +  # Green channel
        depth_image[:, :, 2] * 256**2  # Blue channel
    ) / (256**3 - 1)  # Normalize to range [0, 1]

    # Convert normalized depth values to meters
    depth_in_meters = 1000 * depth_normalized  # Example conversion, adjust as necessary
    
    return depth_in_meters

def main(image_path):
    # Load the depth image
    depth_image = load_depth_image(image_path)
    
    # Extract depth values
    depth_values = extract_depth(depth_image)
    
    # Display the depth values for verification
    print("Depth values (in meters):")
    print(depth_values)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Extract depth values from an RGB depth image.")
    parser.add_argument("image_path", type=str, help="Path to the RGB depth image.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the main function with the provided image path
    main(args.image_path)
